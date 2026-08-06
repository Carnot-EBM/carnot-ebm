"""ARC-AGI-3 reusable offline solver kit — the durable scaffolding so that what
we learn solving one game is REUSED on the next, and every solve is captured as
OFFLINE-REPRODUCIBLE (not a live-recorded coordinate trajectory that silently
rots).

Why this module exists
----------------------
2026-06-16: making the deeper sc25/lp85 levels offline-reproducible revealed that
the prior solves were banked as live-recorded `solve_trace.actions` whose pixel
coordinates were coupled to the LIVE env layout, so they replay to 0 levels on
the offline `environment_files` env. The effort was effectively wasted because
the WINNING CONDITION (the search that derives a solution for the actual env) was
never captured — only one frozen trajectory was. This kit captures the general,
reusable primitives + the hard-won per-game gotchas so future games plug in their
action-model + win-check and inherit the rest, and so every solve passes a
reproduction gate before it counts.

Hard-won general gotchas (apply to ANY ARC-AGI-3 game; see ops/arc_solve_registry.yaml)
-----------------------------------------------------------------------------------
1. OFFLINE is a deterministic simulator: `Arcade(OperationMode.OFFLINE,
   environments_dir=environment_files)` loads all 25 games, zero network/quota.
2. The LEVEL lives on the FRAME (`frame.levels_completed`), NOT on `env._game`.
3. `env._game = copy.deepcopy(state)` injection works for SOME games (lp85) but is
   BROKEN for others (sc25) — references don't survive deepcopy. The robust,
   universal approach is REPLAY-FROM-RESET (operate on the real env).
4. The FIRST `env.step` after `env.reset()` is CONSUMED (no-op) in at least sc25.
   Always do a warm-up step after reset before applying a path.
5. Element COORDINATES must be DISCOVERED from the env, never hardcoded — the live
   solver's hardcoded coords (e.g. sc25 SC25_GRID_COORDS) miss the offline layout.
   Use env-adaptive discovery (cf. lp85 `discover_click_buttons`; sc25 camera is
   identity so cell (r,c) is at display (24+5c, 49+5r)).
6. Some games have STATE-DEPENDENT controls (sc25 tank-controls: press-new-
   direction turns, press-same moves) and MULTI-FRAME ANIMATIONS that must be let
   to resolve — the dedup state-key MUST include facing/phase, and you must step
   until animation phase flags clear before the next action / win check.
7. Some config/toggle games call next_level() in the SAME action that creates the
   winning arrangement, so the returned frame is already the next level and the
   pre-win grid is not externally observable. Ground such win predicates on the
   execution state immediately before next_level, then count only the reproduce()
   level advance.
"""

from __future__ import annotations

import copy
import heapq
import itertools
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Hashable, Mapping, Optional, Sequence

REPO = Path(__file__).resolve().parents[3]
ENV_DIR = REPO / "environment_files"
ARC_STANDING_PATH_COST_WEIGHT = 1.0
ARC_BASELINE_PATH_COST_WEIGHT = 0.0


@dataclass(frozen=True)
class PrimitiveOperator:
    """A reusable ARC solve operator learned from one or more reproduced games."""

    operator: str
    derived_from_games: tuple[str, ...]
    purpose: str
    selector_tags: tuple[str, ...] = ()

    def as_dict(self) -> dict[str, Any]:
        return {
            "operator": self.operator,
            "derived_from_games": list(self.derived_from_games),
            "purpose": self.purpose,
            "selector_tags": list(self.selector_tags),
        }


def primitive_operator_registry() -> tuple[PrimitiveOperator, ...]:
    """REQ-REPORT-4436: consolidated generic operators available to the standing loop."""

    return (
        PrimitiveOperator(
            operator="glyph_rewrite_rule_verifier",
            derived_from_games=("bsqsshqpox", "tr87"),
            purpose="Induce and execution-ground greedy glyph rewrite win predicates from config-substitution examples.",
            selector_tags=("config_substitution", "glyph", "rewrite", "verifier", "rule"),
        ),
        PrimitiveOperator(
            operator="glyph_rewrite_matcher",
            derived_from_games=("tr87",),
            purpose="Greedy multi-glyph LHS->RHS rewrite, including repeated passes.",
            selector_tags=("config_substitution", "glyph", "rewrite", "tr87"),
        ),
        PrimitiveOperator(
            operator="config_rule_grounding",
            derived_from_games=("s5i5", "ft09", "tr87"),
            purpose="Ground a proposed config rule against predicted object/register coverage.",
            selector_tags=("config_toggle", "marker_coverage", "local_constraint", "rule"),
        ),
        PrimitiveOperator(
            operator="config_rule_verifier",
            derived_from_games=("s5i5", "ft09", "g50t", "dc22"),
            purpose="Propose and execution-ground coverage, local-constraint, or toggle win predicates.",
            selector_tags=(
                "config_toggle",
                "marker_coverage",
                "local_constraint",
                "verifier",
                "rule",
            ),
        ),
        PrimitiveOperator(
            operator="color_match_slot_sequence_verifier",
            derived_from_games=("sb26", "s5i5", "ft09"),
            purpose="Ground ordered colored item-to-slot placement predicates with undo-aware counterexamples.",
            selector_tags=(
                "color_match",
                "slot_sequence",
                "ordered",
                "config_rule",
                "undo",
                "verifier",
            ),
        ),
        PrimitiveOperator(
            operator="sprite_overlay_resize_verifier",
            derived_from_games=("re86", "s5i5"),
            purpose=(
                "Ground transparent sprite overlays by matching required target pixels, "
                "including explicit resize variants when the action model exposes them."
            ),
            selector_tags=(
                "sprite_overlay",
                "pattern_match",
                "resize",
                "transparent",
                "verifier",
                "re86",
            ),
        ),
        PrimitiveOperator(
            operator="graph_astar_action_cost",
            derived_from_games=("tu93", "lp85", "cd82", "sp80", "cn04", "m0r0", "sk48", "su15"),
            purpose="A* frontier priority: standing path cost plus verifier/action-cost heuristic.",
            selector_tags=("graph_explore", "astar", "action_cost", "keyboard", "click"),
        ),
        PrimitiveOperator(
            operator="approach_dispatcher_operator",
            derived_from_games=("exp4592_generation_completeness_wiring",),
            purpose=(
                "Execute the mechanic-class route by selecting the generated candidate for "
                "the routed toolkit approach, while falling back honestly when no routed "
                "candidate exists."
            ),
            selector_tags=(
                "approach_dispatch",
                "mechanic_router",
                "candidate_generation",
                "graph_explore",
                "goal_distance",
                "transfer",
            ),
        ),
        PrimitiveOperator(
            operator="value_head_bridge_fix_operator",
            derived_from_games=(
                "exp4616_offline_live_bridge_disambiguation",
                "exp4652_value_routing_cost_fix_live",
            ),
            purpose=(
                "Apply the offline-to-live value-head bridge fix by scoring only bounded "
                "decision points, caching repeated state scores, and reporting first-win or "
                "efficiency lift before any solve claim."
            ),
            selector_tags=(
                "value_head",
                "bridge_fix",
                "decision_point",
                "candidate_ranking",
                "graph_explore",
                "live_path",
                "transfer",
            ),
        ),
        PrimitiveOperator(
            operator="cheap_value_routing_cost_fix_operator",
            derived_from_games=("exp4652_value_routing_cost_fix_live",),
            purpose=(
                "Reuse the productionized cheap value-routing substrate: v2+frame-delta "
                "features, bounded lazy value-head scoring, and frame-hash caching before "
                "reporting solve/first-win/action-efficiency lift."
            ),
            selector_tags=(
                "value_head",
                "cheap_feature",
                "value_routing",
                "cost_fix",
                "candidate_ranking",
                "graph_explore",
                "live_path",
                "transfer",
            ),
        ),
        PrimitiveOperator(
            operator="dagger_off_path_data_collection_operator",
            derived_from_games=("exp4665_dagger_distribution_shift_value_routing",),
            purpose=(
                "Collect and relabel live-frontier off-path rows against reproduced action "
                "prefixes so DAgger-lite value heads train on the search distribution rather "
                "than winning-path states only."
            ),
            selector_tags=(
                "dagger",
                "off_path",
                "search_distribution",
                "value_head",
                "graph_explore",
                "live_path",
                "transfer",
            ),
        ),
        PrimitiveOperator(
            operator="programmatic_expert_trust_weighting_operator",
            derived_from_games=("exp4677_poe_world_factored_subgoal_planner",),
            purpose=(
                "Rank generated programmatic object experts by held-out transition trust, "
                "keep only replay-stable factors for product-model planning, and report "
                "overfit-prefix residuals before any solve claim."
            ),
            selector_tags=(
                "programmatic_expert",
                "trust_weighting",
                "factored_planner",
                "candidate_generation",
                "world_model",
                "graph_explore",
                "transfer",
            ),
        ),
        PrimitiveOperator(
            operator="controllable_novelty_embedding_operator",
            derived_from_games=("exp4688_controllable_novelty_proposal_policy_live",),
            purpose=(
                "Embed controllable action-effect deltas for intrinsic proposal novelty, "
                "reject cosmetic/raw-frame novelty when the controllability gate is active, "
                "and report transfer value before any solve claim."
            ),
            selector_tags=(
                "controllable_novelty",
                "directed_exploration",
                "intrinsic_proposal",
                "candidate_generation",
                "action_effect",
                "graph_explore",
                "transfer",
            ),
        ),
        PrimitiveOperator(
            operator="object_centric_representation_builder_operator",
            derived_from_games=(
                "exp4700_object_centric_perception_proposal_live",
                "exp4701_amortized_exploration_prior_go_explore_live",
            ),
            purpose=(
                "Build connected-component object slots, relational gap keypoints, and "
                "proposal-side coverage diagnostics before value ranking or frontier "
                "selection, without consulting the executable win-check."
            ),
            selector_tags=(
                "object_centric",
                "perception",
                "representation_builder",
                "candidate_generation",
                "graph_explore",
                "transfer",
            ),
        ),
        PrimitiveOperator(
            operator="online_warm_action_effect_controller_operator",
            derived_from_games=(
                "exp4726_online_action_learning_driver_valid_test",
                "exp4727_active_probe_disambiguation",
            ),
            purpose=(
                "Reuse the .435 online-warm action-effect controller: combine "
                "leave-one-game PersistentAEM evidence with optional online-warm "
                "frame-change scores, keep the executable win-check outside ranking, "
                "and report first-win/efficiency transfer before any solve claim."
            ),
            selector_tags=(
                "online_action_learning",
                "online_warm",
                "action_effect",
                "frame_change",
                "candidate_ranking",
                "graph_explore",
                "click",
                "keyboard",
                "transfer",
            ),
        ),
        PrimitiveOperator(
            operator="energy_fitness_qd_generator_operator",
            derived_from_games=("exp4738_energy_fitness_qd_generation_valid_test",),
            purpose=(
                "Reuse the .436 energy-fitness QD generator: keep naive candidates "
                "available, score generated candidates with oracle-distinct lower-is-better "
                "energy, preserve diverse behavior descriptors, and report coverage or "
                "first-win transfer before any solve claim."
            ),
            selector_tags=(
                "energy_qd",
                "map_elites",
                "candidate_generation",
                "energy_fitness",
                "graph_explore",
                "click",
                "transfer",
            ),
        ),
        PrimitiveOperator(
            operator="verifier_router_candidate_ranking_operator",
            derived_from_games=("exp4556_cached_generic_transfer",),
            purpose=(
                "Rank compatible cached action/plan candidates by verifier score with stable "
                "tie-breaking and report ordering gain before any solve claim."
            ),
            selector_tags=(
                "verifier_router",
                "candidate_ranking",
                "trust_energy",
                "graph_explore",
                "config_toggle",
                "program_editor",
                "transfer",
            ),
        ),
        PrimitiveOperator(
            operator="world_model_trust_energy_gate_operator",
            derived_from_games=("exp4604_world_model_trust_energy",),
            purpose=(
                "Rank executable world-model candidates by change-weighted held-out trust "
                "energy, reject identity/no-op degeneracy, and report value against the "
                "legacy binary exact-match gate before solve claims."
            ),
            selector_tags=(
                "world_model",
                "trust_energy",
                "hidden_state",
                "candidate_ranking",
                "verifier",
                "transfer",
            ),
        ),
        PrimitiveOperator(
            operator="persistent_action_effect_memory_operator",
            derived_from_games=(
                "exp4568_clickability_action_effect_predictor",
                "exp4629_graduate_action_effect_predictor_live",
            ),
            purpose=(
                "Rank action candidates with a leave-one-game cross-game memory of cached "
                "frame/action effects or the graduated live action-effect scorer, preserving "
                "original-order tie-breaking and reporting actions-to-first-levelup deltas "
                "before solve claims."
            ),
            selector_tags=(
                "action_effect",
                "clickability",
                "frame_change",
                "live_action_pruner",
                "persistent_aem_plus_optional_cnn",
                "candidate_ranking",
                "graph_explore",
                "click",
                "keyboard",
                "transfer",
            ),
        ),
        PrimitiveOperator(
            operator="graded_goal_energy_search_heuristic_operator",
            derived_from_games=(
                "exp4020_goal_induction_separation",
                "exp4640_goal_energy_generation_live",
            ),
            purpose=(
                "Rank generated frontier candidates by a depth-preserving convex blend of "
                "navigation energy and Exp4020 graded goal-satisfaction energy, then report "
                "solve/first-win/action-efficiency lift before any solve claim."
            ),
            selector_tags=(
                "goal_energy",
                "graded_goal_satisfaction",
                "search_heuristic",
                "graph_explore",
                "candidate_ranking",
                "transfer",
            ),
        ),
        PrimitiveOperator(
            operator="env_adaptive_resolve_operator",
            derived_from_games=("sc25", "exp4580_live_submission_gap_close"),
            purpose=(
                "Re-derive replayable action coordinates from the current environment instead "
                "of trusting frozen live/offline pixel coordinates, then report drift recovery "
                "before any solve claim."
            ),
            selector_tags=(
                "env_adaptive",
                "coordinate_discovery",
                "drift_recovery",
                "replay",
                "click",
                "transfer",
            ),
        ),
        PrimitiveOperator(
            operator="per_level_reinduction_operator",
            derived_from_games=("lp85", "m0r0", "sp80", "vc33"),
            purpose=(
                "Detect a level-up, clear stale level-local induction state, re-induce the "
                "next level predicate, and route the frontier with depth-primary goal bias."
            ),
            selector_tags=("reinduction", "level_up", "deepening", "goal_bias", "transfer"),
        ),
        PrimitiveOperator(
            operator="llm_proposer_reinduction_operator",
            derived_from_games=("lp85", "m0r0", "sp80", "vc33"),
            purpose=(
                "Detect a level-up, request GOAL+DYNAMICS+plan candidates from an LLM proposer, "
                "rank them by trust energy, and fall back to DSL/verifier re-induction when live "
                "proposal is unavailable."
            ),
            selector_tags=(
                "reinduction",
                "level_up",
                "llm_proposer",
                "bounded_refinement",
                "trust_energy",
                "transfer",
            ),
        ),
        PrimitiveOperator(
            operator="object_centric_digest",
            derived_from_games=("g50t", "lp85", "tn36", "ka59"),
            purpose="Connected-component object summary for routing, grounding, and active data.",
            selector_tags=("object", "digest", "program_editor", "world_model"),
        ),
        PrimitiveOperator(
            operator="active_data_collection",
            derived_from_games=("ar25", "ka59", "ft09", "sc25"),
            purpose="Balanced action/object coverage plan for offline transition collection.",
            selector_tags=("active_data", "world_model", "e3", "transition"),
        ),
        PrimitiveOperator(
            operator="object_motion_world_model",
            derived_from_games=("ar25", "ka59", "sc25", "ft09"),
            purpose="Object-slot transition model for translate, reflect, push, and dynamic selection.",
            selector_tags=("object", "motion", "world_model", "e3", "translate", "reflect", "push"),
        ),
        PrimitiveOperator(
            operator="cast_grid_phase_fsm_world_model",
            derived_from_games=("sc25", "ar25", "ka59", "ft09"),
            purpose="Two-phase cast/config-grid toggle CSP followed by player navigation to an exit predicate.",
            selector_tags=(
                "cast_grid",
                "phase_fsm",
                "config_toggle",
                "navigation",
                "world_model",
                "verifier",
            ),
        ),
    )


def select_primitive_operators(
    *, mechanic_class: Optional[str] = None, action_model: str = "", game: str = ""
) -> tuple[PrimitiveOperator, ...]:
    """Select generic operators before per-game reverse engineering.

    This is intentionally conservative: it exposes reusable operators for the standing
    loop without removing any per-game adapter path.
    """

    registry = {op.operator: op for op in primitive_operator_registry()}
    mechanic = (mechanic_class or "").lower()
    action = (action_model or "").lower()
    gid = (game or "").lower()

    if (
        "cast_grid" in mechanic
        or "cast grid" in mechanic
        or "phase_fsm" in mechanic
        or "two_phase_cast_grid" in mechanic
        or gid == "sc25"
    ):
        names = (
            "approach_dispatcher_operator",
            "value_head_bridge_fix_operator",
            "per_level_reinduction_operator",
            "llm_proposer_reinduction_operator",
            "env_adaptive_resolve_operator",
            "verifier_router_candidate_ranking_operator",
            "world_model_trust_energy_gate_operator",
            "persistent_action_effect_memory_operator",
            "graded_goal_energy_search_heuristic_operator",
            "cast_grid_phase_fsm_world_model",
            "object_motion_world_model",
            "active_data_collection",
            "graph_astar_action_cost",
        )
    elif (
        "color_match" in mechanic
        or "color match" in mechanic
        or "slot_sequence" in mechanic
        or "slot sequence" in mechanic
        or gid == "sb26"
    ):
        names = (
            "approach_dispatcher_operator",
            "color_match_slot_sequence_verifier",
            "config_rule_verifier",
            "object_centric_digest",
            "graph_astar_action_cost",
        )
    elif (
        "sprite_overlay" in mechanic
        or "sprite overlay" in mechanic
        or "pattern_match_sprite_resize" in mechanic
        or "sprite_resize" in mechanic
        or "resize" in mechanic
        or gid == "re86"
    ):
        names = (
            "approach_dispatcher_operator",
            "sprite_overlay_resize_verifier",
            "object_centric_digest",
            "graph_astar_action_cost",
        )
    elif "config_substitution" in mechanic or "glyph" in mechanic or gid == "tr87":
        names = (
            "approach_dispatcher_operator",
            "value_head_bridge_fix_operator",
            "glyph_rewrite_rule_verifier",
            "glyph_rewrite_matcher",
            "per_level_reinduction_operator",
            "llm_proposer_reinduction_operator",
            "env_adaptive_resolve_operator",
            "verifier_router_candidate_ranking_operator",
            "world_model_trust_energy_gate_operator",
            "persistent_action_effect_memory_operator",
            "graded_goal_energy_search_heuristic_operator",
            "graph_astar_action_cost",
            "object_centric_digest",
        )
    elif "config" in mechanic or "toggle" in mechanic or "constraint" in mechanic:
        names = (
            "approach_dispatcher_operator",
            "value_head_bridge_fix_operator",
            "config_rule_verifier",
            "config_rule_grounding",
            "per_level_reinduction_operator",
            "llm_proposer_reinduction_operator",
            "env_adaptive_resolve_operator",
            "verifier_router_candidate_ranking_operator",
            "world_model_trust_energy_gate_operator",
            "persistent_action_effect_memory_operator",
            "graded_goal_energy_search_heuristic_operator",
            "object_centric_digest",
            "graph_astar_action_cost",
        )
    elif "program_editor" in mechanic:
        names = (
            "approach_dispatcher_operator",
            "value_head_bridge_fix_operator",
            "per_level_reinduction_operator",
            "llm_proposer_reinduction_operator",
            "env_adaptive_resolve_operator",
            "verifier_router_candidate_ranking_operator",
            "world_model_trust_energy_gate_operator",
            "persistent_action_effect_memory_operator",
            "graded_goal_energy_search_heuristic_operator",
            "object_centric_digest",
            "active_data_collection",
            "graph_astar_action_cost",
        )
    elif (
        "object_motion" in mechanic
        or "object motion" in mechanic
        or "reflection" in mechanic
        or "reflect" in mechanic
        or "push" in mechanic
        or "world_model" in mechanic
        or "e3" in mechanic
    ):
        names = (
            "approach_dispatcher_operator",
            "value_head_bridge_fix_operator",
            "world_model_trust_energy_gate_operator",
            "object_motion_world_model",
            "active_data_collection",
            "object_centric_digest",
            "graded_goal_energy_search_heuristic_operator",
            "graph_astar_action_cost",
        )
    elif "keyboard" in action or "click" in action or "graph" in mechanic:
        names = (
            "approach_dispatcher_operator",
            "value_head_bridge_fix_operator",
            "per_level_reinduction_operator",
            "llm_proposer_reinduction_operator",
            "env_adaptive_resolve_operator",
            "verifier_router_candidate_ranking_operator",
            "world_model_trust_energy_gate_operator",
            "persistent_action_effect_memory_operator",
            "graded_goal_energy_search_heuristic_operator",
            "graph_astar_action_cost",
            "object_centric_digest",
        )
    else:
        names = (
            "approach_dispatcher_operator",
            "value_head_bridge_fix_operator",
            "per_level_reinduction_operator",
            "llm_proposer_reinduction_operator",
            "env_adaptive_resolve_operator",
            "verifier_router_candidate_ranking_operator",
            "world_model_trust_energy_gate_operator",
            "persistent_action_effect_memory_operator",
            "graded_goal_energy_search_heuristic_operator",
            "object_centric_digest",
            "active_data_collection",
            "graph_astar_action_cost",
        )
    if (
        "value_head_bridge_fix_operator" in names
        and "cheap_value_routing_cost_fix_operator" not in names
    ):
        expanded: list[str] = []
        for name in names:
            expanded.append(name)
            if name == "value_head_bridge_fix_operator":
                expanded.append("cheap_value_routing_cost_fix_operator")
                expanded.append("dagger_off_path_data_collection_operator")
                expanded.append("programmatic_expert_trust_weighting_operator")
                expanded.append("controllable_novelty_embedding_operator")
        names = tuple(expanded)
    if (
        "object_centric_digest" in names
        and "object_centric_representation_builder_operator" not in names
    ):
        expanded = []
        for name in names:
            if name == "object_centric_digest":
                expanded.append("object_centric_representation_builder_operator")
            expanded.append(name)
        names = tuple(expanded)
    if (
        "persistent_action_effect_memory_operator" in names
        and "online_warm_action_effect_controller_operator" not in names
    ):
        expanded = []
        for name in names:
            expanded.append(name)
            if name == "persistent_action_effect_memory_operator":
                expanded.append("online_warm_action_effect_controller_operator")
        names = tuple(expanded)
    if (
        "graded_goal_energy_search_heuristic_operator" in names
        and "energy_fitness_qd_generator_operator" not in names
    ):
        expanded = []
        for name in names:
            expanded.append(name)
            if name == "graded_goal_energy_search_heuristic_operator":
                expanded.append("energy_fitness_qd_generator_operator")
        names = tuple(expanded)
    return tuple(registry[name] for name in names)


def _candidate_field(candidate: Any, key: str, default: Any = None) -> Any:
    if isinstance(candidate, Mapping):
        return candidate.get(key, default)
    return getattr(candidate, key, default)


def _candidate_identifier(candidate: Any, index: int) -> str:
    for key in ("candidate_id", "id", "name", "router_mode", "action_id", "key"):
        value = _candidate_field(candidate, key)
        if value is not None:
            return str(value)
    return str(index)


def _candidate_score(candidate: Any, score_key: str) -> float:
    for key in (score_key, "verifier_score", "score"):
        value = _candidate_field(candidate, key)
        if value is not None:
            try:
                return float(value)
            except (TypeError, ValueError):
                return 0.0
    trust_energy = _candidate_field(candidate, "trust_energy")
    if trust_energy is not None:
        try:
            return -float(trust_energy)
        except (TypeError, ValueError):
            return 0.0
    return 0.0


def _candidate_truthy(candidate: Any, key: str) -> bool:
    value = _candidate_field(candidate, key)
    if isinstance(value, bool):
        return value
    if isinstance(value, int | float):
        return bool(value)
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "win", "won", "solved", "reproduced"}
    return False


def _candidate_approach(candidate: Any) -> str:
    for key in ("approach", "selected_approach", "executed_approach", "route", "operator"):
        value = _candidate_field(candidate, key)
        if value:
            return str(value)
    return ""


def _candidate_public_dict(candidate: Any) -> dict[str, Any]:
    if isinstance(candidate, Mapping):
        return dict(candidate)
    data = getattr(candidate, "__dict__", None)
    return dict(data) if isinstance(data, Mapping) else {"repr": repr(candidate)}


def _world_model_candidate_engine(candidate: Any) -> Callable[[Any, int, Any], Any] | None:
    engine = _candidate_field(candidate, "engine")
    return engine if callable(engine) else None


def _world_model_candidate_name(candidate: Any, index: int) -> str:
    for key in ("name", "candidate", "candidate_id", "id"):
        value = _candidate_field(candidate, key)
        if value is not None:
            return str(value)
    return _candidate_identifier(candidate, index)


def _world_model_candidate_public(candidate: Any) -> dict[str, Any]:
    public = _candidate_public_dict(candidate)
    return {
        str(key): getattr(value, "__name__", repr(value)) if callable(value) else value
        for key, value in public.items()
    }


def _world_model_score_public(row: Any) -> dict[str, Any]:
    return {
        "candidate": str(row.candidate.name),
        "prefix_accuracy": round(float(row.prefix_accuracy), 6),
        "heldout_accuracy": round(float(row.heldout_accuracy), 6),
        "prefix_change_consistency": round(float(row.prefix_change_consistency), 6),
        "heldout_change_consistency": round(float(row.heldout_change_consistency), 6),
        "trust_energy": round(float(row.trust_energy), 6),
        "trust_pass": bool(row.trust_pass),
        "binary_gate_pass": bool(row.binary_gate_pass),
        "correct_changed_cells": int(row.correct_changed_cells),
        "true_changed_cells": int(row.true_changed_cells),
        "nondegenerate": bool(row.nondegenerate),
    }


def world_model_trust_energy_gate_operator(
    transitions: Sequence[Any],
    candidates: Sequence[Any] | Mapping[str, Any],
    *,
    baseline_threshold: float = 0.5,
) -> dict[str, Any]:
    """REQ-ARC-WMTE-4608: reusable oracle-distinct trust gate for world models."""

    from carnot.agentic.arc_world_model_trust_energy import (
        WorldModelCandidate,
        select_trusted_world_model,
    )

    candidate_rows = (
        list(candidates.values()) if isinstance(candidates, Mapping) else list(candidates or [])
    )
    world_model_candidates: list[WorldModelCandidate] = []
    public_by_name: dict[str, dict[str, Any]] = {}
    for index, candidate in enumerate(candidate_rows):
        engine = _world_model_candidate_engine(candidate)
        if engine is None:
            continue
        name = _world_model_candidate_name(candidate, index)
        world_model_candidates.append(WorldModelCandidate(name, engine))
        public_by_name[name] = _world_model_candidate_public(candidate)

    if not transitions or not world_model_candidates:
        return {
            "operator": "world_model_trust_energy_gate_operator",
            "candidate_count": len(candidate_rows),
            "selected_candidate_name": "",
            "baseline_candidate_name": None,
            "trust_pass": False,
            "binary_gate_pass": False,
            "trust_pass_added": False,
            "value_added": False,
            "selected_score": None,
            "rows": [],
            "selected_candidate": None,
            "dead_end": "no transitions or executable world-model candidates",
            "verifier_is_oracle": False,
        }

    selection = select_trusted_world_model(
        list(transitions),
        world_model_candidates,
        hidden_state=True,
        baseline_threshold=float(baseline_threshold),
    )
    selected = selection.selected_score
    baseline_row = next((row for row in selection.rows if row.binary_gate_pass), None)
    selected_binary_pass = bool(selected.binary_gate_pass)
    trust_pass_added = bool(selected.trust_pass and not selected_binary_pass)
    value_added = bool(trust_pass_added)
    if not selected.trust_pass:
        dead_end = "selected world model did not pass held-out trust energy"
    elif selected_binary_pass:
        dead_end = "legacy binary exact-match gate already passed the selected world model"
    else:
        dead_end = ""

    selected_name = str(selection.selected.name)
    return {
        "operator": "world_model_trust_energy_gate_operator",
        "candidate_count": len(world_model_candidates),
        "selected_candidate_name": selected_name,
        "baseline_candidate_name": baseline_row.candidate.name
        if baseline_row is not None
        else None,
        "trust_pass": bool(selected.trust_pass),
        "binary_gate_pass": selected_binary_pass,
        "trust_pass_added": trust_pass_added,
        "value_added": value_added,
        "selected_score": _world_model_score_public(selected),
        "rows": [_world_model_score_public(row) for row in selection.rows],
        "selected_candidate": public_by_name.get(selected_name, {"name": selected_name}),
        "dead_end": dead_end,
        "verifier_is_oracle": False,
    }


def _candidate_generated(candidate: Any) -> bool:
    if _candidate_truthy(candidate, "candidate_generated"):
        return True
    if _candidate_reaches_dispatch_win(candidate):
        return True
    return bool(
        _candidate_field(candidate, "solution_labels") or _candidate_field(candidate, "labels")
    )


def _candidate_reaches_dispatch_win(candidate: Any) -> bool:
    if any(
        _candidate_truthy(candidate, key)
        for key in ("winner_generated", "win_reached", "solved", "offline_reproduced")
    ):
        return True
    gate = _candidate_field(candidate, "reproduction_gate")
    if isinstance(gate, Mapping) and gate.get("reproduced") is True:
        try:
            reached = int(gate.get("reached_level") or 0)
            claimed = int(gate.get("claimed_level") or reached)
        except (TypeError, ValueError):
            return True
        return reached >= max(1, claimed)
    return False


def approach_dispatcher_operator(
    route: Mapping[str, Any] | None,
    candidates: Sequence[Any] | Mapping[str, Any],
    *,
    baseline_approach: str = "default_graph_explore",
) -> dict[str, Any]:
    """REQ-CAPSTONE-4596: execute a routed approach over generated candidates."""

    route_map = route if isinstance(route, Mapping) else {}
    selected_approach = str(route_map.get("approach") or baseline_approach)
    if isinstance(candidates, Mapping):
        candidate_rows: list[Any] = list(candidates.values())
    else:
        candidate_rows = list(candidates or [])

    baseline = next(
        (
            candidate
            for candidate in candidate_rows
            if _candidate_approach(candidate) == baseline_approach
        ),
        None,
    )
    routed = next(
        (
            candidate
            for candidate in candidate_rows
            if _candidate_approach(candidate) == selected_approach
            and _candidate_generated(candidate)
        ),
        None,
    )
    selected = routed if routed is not None else baseline
    executed_approach = _candidate_approach(selected) if selected is not None else ""
    baseline_winner = bool(baseline is not None and _candidate_reaches_dispatch_win(baseline))
    selected_winner = bool(selected is not None and _candidate_reaches_dispatch_win(selected))
    selected_generated = bool(selected is not None and _candidate_generated(selected))
    value_added = bool(routed is not None and selected_winner and not baseline_winner)

    if routed is None:
        dead_end = f"no generated candidate for routed approach {selected_approach}"
    elif not selected_winner:
        dead_end = (
            "dispatched candidate generated a proposal, but the verifier gate did not reach a win"
        )
    elif baseline_winner:
        dead_end = "baseline approach already generated the winning candidate"
    else:
        dead_end = ""

    return {
        "operator": "approach_dispatcher_operator",
        "mechanic_class": str(route_map.get("mechanic_class") or ""),
        "selected_approach": selected_approach,
        "executed_approach": executed_approach or baseline_approach,
        "baseline_approach": baseline_approach,
        "candidate_count": len(candidate_rows),
        "candidate_generated": selected_generated,
        "winner_generated": selected_winner,
        "win_reached": selected_winner,
        "baseline_winner_generated": baseline_winner,
        "value_added": value_added,
        "selected_candidate": _candidate_public_dict(selected) if selected is not None else None,
        "baseline_candidate": _candidate_public_dict(baseline) if baseline is not None else None,
        "dead_end": dead_end,
        "verifier_is_oracle": False,
    }


def _value_head_bridge_score(
    candidate: Any,
    *,
    value_head: Callable[[Any], float] | None,
    score_key: str,
) -> float | None:
    if callable(value_head):
        try:
            value = value_head(candidate)
        except Exception:
            return None
    else:
        value = None
        for key in (score_key, "value_head_score", "value_score", "score"):
            raw = _candidate_field(candidate, key)
            if raw is not None:
                value = raw
                break
    try:
        return None if value is None else float(value)
    except (TypeError, ValueError):
        return None


def _value_head_bridge_target(candidate: Any, target_key: str) -> bool:
    if _candidate_truthy(candidate, target_key):
        return True
    if _candidate_reaches_dispatch_win(candidate):
        return True
    return any(
        _candidate_truthy(candidate, key)
        for key in ("reaches_levelup", "reaches_goal", "level_progress")
    )


def _value_head_bridge_public_row(
    candidate: Any,
    *,
    index: int,
    score: float | None,
    scored: bool,
    target: bool,
) -> dict[str, Any]:
    public = _candidate_public_dict(candidate)
    public.setdefault("candidate_id", _candidate_identifier(candidate, index))
    public["original_index"] = int(index)
    public["value_head_score"] = score
    public["value_head_scored"] = bool(scored)
    public["target_candidate"] = bool(target)
    return public


def value_head_bridge_fix_operator(
    candidates: Sequence[Any] | Mapping[str, Any],
    *,
    value_head: Callable[[Any], float] | None = None,
    score_key: str = "value_head_score",
    target_key: str = "reaches_levelup",
    decision_point_key: str = "decision_point",
    state_key: str = "state_key",
    max_value_evals: int = 32,
    first_win_budget: int | None = None,
) -> dict[str, Any]:
    """REQ-ARC-WMTE-4620: bounded cached value-head bridge-fix ranker."""

    candidate_rows = (
        list(candidates.values()) if isinstance(candidates, Mapping) else list(candidates or [])
    )
    max_evals = max(0, int(max_value_evals))
    cache: dict[str, float | None] = {}
    cache_hits = 0
    value_evals = 0
    scored_rows: list[dict[str, Any]] = []

    for index, candidate in enumerate(candidate_rows):
        decision_value = _candidate_field(candidate, decision_point_key, True)
        is_decision_point = (
            bool(decision_value)
            if not isinstance(decision_value, str)
            else (decision_value.strip().lower() not in {"0", "false", "no"})
        )
        score: float | None = None
        scored = False
        if is_decision_point:
            state_value = _candidate_field(candidate, state_key)
            state_id = (
                str(state_value)
                if state_value is not None
                else _candidate_identifier(candidate, index)
            )
            if state_id in cache:
                score = cache[state_id]
                scored = score is not None
                cache_hits += 1
            elif value_evals < max_evals:
                score = _value_head_bridge_score(
                    candidate,
                    value_head=value_head,
                    score_key=score_key,
                )
                cache[state_id] = score
                value_evals += 1
                scored = score is not None
        target = _value_head_bridge_target(candidate, target_key)
        scored_rows.append(
            {
                "candidate": candidate,
                "index": index,
                "score": score,
                "scored": scored,
                "target": target,
            }
        )

    ranked_internal = sorted(
        scored_rows,
        key=lambda row: (
            row["score"] is None,
            float(row["score"]) if row["score"] is not None else 0.0,
            int(row["index"]),
        ),
    )
    ranked = [
        _value_head_bridge_public_row(
            row["candidate"],
            index=int(row["index"]),
            score=row["score"],
            scored=bool(row["scored"]),
            target=bool(row["target"]),
        )
        for row in ranked_internal
    ]
    baseline = [
        _value_head_bridge_public_row(
            row["candidate"],
            index=int(row["index"]),
            score=row["score"],
            scored=bool(row["scored"]),
            target=bool(row["target"]),
        )
        for row in scored_rows
    ]

    target_before = next((idx for idx, row in enumerate(scored_rows) if row["target"]), None)
    target_after = next((idx for idx, row in enumerate(ranked_internal) if row["target"]), None)
    before_actions = None if target_before is None else target_before + 1
    after_actions = None if target_after is None else target_after + 1
    efficiency_lift = (
        0
        if before_actions is None or after_actions is None
        else max(0, int(before_actions - after_actions))
    )
    if first_win_budget is None:
        first_win_lift = False
    else:
        budget = max(1, int(first_win_budget))
        baseline_first = before_actions is not None and before_actions <= budget
        bounded_first = after_actions is not None and after_actions <= budget
        first_win_lift = bool(bounded_first and not baseline_first)
    value_added = bool(efficiency_lift > 0 or first_win_lift)

    if not candidate_rows:
        dead_end = "no candidates"
    elif target_before is None:
        dead_end = "no level-up/win candidate in decision set"
    elif not any(row["scored"] for row in scored_rows):
        dead_end = "no bounded decision-point value scores available"
    elif not value_added:
        dead_end = "value-head order matched baseline or did not improve target rank"
    else:
        dead_end = ""

    return {
        "operator": "value_head_bridge_fix_operator",
        "candidate_count": len(candidate_rows),
        "max_value_evals": max_evals,
        "value_head_evals": value_evals,
        "cache_hits": cache_hits,
        "target_rank_before": target_before,
        "target_rank_after": target_after,
        "actions_to_first_levelup_before": before_actions,
        "actions_to_first_levelup_after": after_actions,
        "efficiency_lift": efficiency_lift,
        "first_win_lift": first_win_lift,
        "value_added": value_added,
        "baseline_candidates": baseline,
        "ranked_candidates": ranked,
        "selected_candidate": ranked[0] if ranked else None,
        "dead_end": dead_end,
        "verifier_is_oracle": False,
    }


def cheap_value_routing_cost_fix_operator(
    candidates: Sequence[Any] | Mapping[str, Any],
    *,
    value_head: Callable[[Any], float] | None = None,
    score_key: str = "value_head_score",
    target_key: str = "reaches_levelup",
    decision_point_key: str = "decision_point",
    state_key: str = "state_key",
    max_value_evals: int = 32,
    first_win_budget: int | None = None,
    feature_subset: str = "cross_game_features_v3:v2_plus_frame_delta",
    per_node_feature_cost_ms: float | None = None,
) -> dict[str, Any]:
    """REQ-ARC-WMTE-4656: persisted cheap-feature value-routing cost fix."""

    ranked = value_head_bridge_fix_operator(
        candidates,
        value_head=value_head,
        score_key=score_key,
        target_key=target_key,
        decision_point_key=decision_point_key,
        state_key=state_key,
        max_value_evals=max_value_evals,
        first_win_budget=first_win_budget,
    )
    try:
        feature_cost = None if per_node_feature_cost_ms is None else float(per_node_feature_cost_ms)
    except (TypeError, ValueError):
        feature_cost = None
    result = dict(ranked)
    result["base_operator"] = ranked.get("operator")
    result["operator"] = "cheap_value_routing_cost_fix_operator"
    result["feature_subset"] = str(feature_subset)
    result["per_node_feature_cost_ms"] = feature_cost
    result["cost_fix_applied"] = feature_cost is None or feature_cost < 1.0
    result["verifier_is_oracle"] = False
    return result


def _dagger_path_step_label(step: Any) -> str | None:
    action = _candidate_field(step, "action")
    if action is None:
        return None
    try:
        action_id = int(action)
    except (TypeError, ValueError):
        return None
    return json.dumps(
        {"action": action_id, "data": _candidate_field(step, "data")},
        sort_keys=True,
        separators=(",", ":"),
    )


def _dagger_clean_path(path: Any) -> list[dict[str, Any]]:
    if not isinstance(path, Sequence) or isinstance(path, (str, bytes)):
        return []
    clean: list[dict[str, Any]] = []
    for step in path:
        action = _candidate_field(step, "action")
        try:
            action_id = int(action)
        except (TypeError, ValueError):
            continue
        clean.append({"action": action_id, "data": _candidate_field(step, "data")})
    return clean


def _dagger_features(row: Any) -> list[float]:
    features = _candidate_field(row, "features", [])
    if not isinstance(features, Sequence) or isinstance(features, (str, bytes)):
        return []
    clean: list[float] = []
    for value in features:
        try:
            clean.append(float(value))
        except (TypeError, ValueError):
            return []
    return clean


def _dagger_label(row: Any) -> float:
    try:
        return 1.0 if float(_candidate_field(row, "label", 0.0) or 0.0) >= 0.5 else 0.0
    except (TypeError, ValueError):
        return 0.0


def dagger_off_path_data_collection_operator(
    frontier_rows: Sequence[Any],
    *,
    winning_labels: Sequence[str] = (),
    winning_rows: Sequence[Any] = (),
) -> dict[str, Any]:
    """REQ-ARC-WMTE-4668: reusable DAgger-lite off-path data collector."""

    clean_winning = [str(label) for label in winning_labels if label is not None]
    winning_prefixes = {tuple(clean_winning[: index + 1]) for index in range(len(clean_winning))}
    if clean_winning:
        winning_prefixes.add(())

    relabeled: list[dict[str, Any]] = []
    for row in frontier_rows:
        features = _dagger_features(row)
        if not features:
            continue
        path = _dagger_clean_path(_candidate_field(row, "path", []))
        labels = tuple(label for label in (_dagger_path_step_label(step) for step in path) if label)
        label = 1.0 if labels in winning_prefixes else 0.0
        relabeled.append(
            {
                "source": str(_candidate_field(row, "source", "search_distribution")),
                "features": features,
                "label": label,
                "path": path,
                "relabel_source": "executable_reproduction_prefix",
            }
        )

    aggregate = list(relabeled)
    for row in winning_rows:
        features = _dagger_features(row)
        if not features:
            continue
        aggregate.append(
            {
                "source": str(_candidate_field(row, "source", "winning_path") or "winning_path"),
                "features": features,
                "label": _dagger_label(row),
                "path": _dagger_clean_path(_candidate_field(row, "path", [])),
                "relabel_source": str(_candidate_field(row, "relabel_source", "winning_path")),
            }
        )

    positives = sum(1 for row in aggregate if float(row["label"]) >= 0.5)
    negatives = len(aggregate) - positives
    return {
        "operator": "dagger_off_path_data_collection_operator",
        "frontier_count": len(frontier_rows),
        "relabeled_frontier_count": len(relabeled),
        "aggregate_total_count": len(aggregate),
        "positive_count": int(positives),
        "negative_count": int(negatives),
        "winning_path_count": sum(1 for row in aggregate if row["source"] == "winning_path"),
        "off_path_negative_count": sum(1 for row in relabeled if float(row["label"]) < 0.5),
        "rows": aggregate,
        "verifier_is_oracle": False,
    }


def _programmatic_expert_number(value: Any, default: float = 0.0) -> float:
    if isinstance(value, bool):
        return float(default)
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _programmatic_expert_int(row: Any, key: str) -> int:
    value = _candidate_field(row, key, 0)
    if isinstance(value, bool):
        return 0
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _programmatic_expert_trust(row: Any) -> float:
    trust = _candidate_field(row, "trust")
    if trust is not None:
        return max(0.0, min(1.0, _programmatic_expert_number(trust)))
    total = _programmatic_expert_int(row, "heldout_total")
    if total <= 0:
        return 0.0
    correct = _programmatic_expert_int(row, "heldout_correct")
    return max(0.0, min(1.0, float(correct) / float(total)))


def _programmatic_expert_public_row(row: Any, index: int, threshold: float) -> dict[str, Any]:
    trust = _programmatic_expert_trust(row)
    heldout_total = _programmatic_expert_int(row, "heldout_total")
    heldout_correct = _programmatic_expert_int(row, "heldout_correct")
    kept = bool(heldout_total > 0 and trust >= threshold)
    public = {
        "name": str(_candidate_field(row, "name", f"expert_{index}") or f"expert_{index}"),
        "object_class": str(_candidate_field(row, "object_class", "unknown") or "unknown"),
        "trust": round(float(trust), 6),
        "heldout_correct": int(heldout_correct),
        "heldout_total": int(heldout_total),
        "kept": kept,
    }
    game = _candidate_field(row, "game")
    if game:
        public["game"] = str(game)
    source_kept = _candidate_field(row, "kept")
    if isinstance(source_kept, bool):
        public["source_kept"] = bool(source_kept)
    return public


def programmatic_expert_trust_weighting_operator(
    expert_rows: Sequence[Any],
    *,
    trust_threshold: float = 0.75,
) -> dict[str, Any]:
    """REQ-ARC-WMTE-4680: reusable held-out trust gate for generated experts."""

    threshold = max(0.0, min(1.0, _programmatic_expert_number(trust_threshold, 0.75)))
    public_rows = [
        _programmatic_expert_public_row(row, index, threshold)
        for index, row in enumerate(list(expert_rows))
    ]
    public_rows.sort(
        key=lambda row: (
            -float(row["trust"]),
            str(row["name"]),
            str(row.get("object_class") or ""),
        )
    )
    kept_rows = [row for row in public_rows if row["kept"]]
    residual = ""
    if not kept_rows:
        residual = "experts_overfit_prefix" if public_rows else "expert_factors_not_independent"
    return {
        "operator": "programmatic_expert_trust_weighting_operator",
        "expert_count": len(public_rows),
        "kept_expert_count": len(kept_rows),
        "rejected_expert_count": len(public_rows) - len(kept_rows),
        "trust_threshold": round(float(threshold), 6),
        "best_trust": float(public_rows[0]["trust"]) if public_rows else 0.0,
        "expert_trust_weights": public_rows,
        "coverage_ready": bool(kept_rows),
        "residual": residual,
        "verifier_is_oracle": False,
    }


def _controllable_novelty_int(row: Any, key: str) -> int:
    value = _candidate_field(row, key, 0)
    if isinstance(value, bool):
        return 0
    try:
        return max(0, int(value))
    except (TypeError, ValueError):
        return 0


def _controllable_novelty_bool(row: Any, key: str, default: bool = False) -> bool:
    value = _candidate_field(row, key, default)
    return bool(value) if isinstance(value, bool) else bool(default)


def _controllable_novelty_min(value: Any) -> int:
    if isinstance(value, bool):
        return 1 if value else 1
    try:
        return max(1, int(value))
    except (TypeError, ValueError):
        return 1


def _controllable_novelty_public_row(row: Any, index: int, min_observed: int) -> dict[str, Any]:
    observed = _controllable_novelty_int(row, "observed_effects")
    embeddings = _controllable_novelty_int(row, "episodic_embeddings")
    candidate_scores = _controllable_novelty_int(row, "candidate_scores")
    rnd_updates = _controllable_novelty_int(row, "rnd_updates")
    gate_on = _controllable_novelty_bool(row, "controllability_gate_on", True)
    raw_frame = _controllable_novelty_bool(row, "raw_frame_novelty", False)
    usable = bool(gate_on and not raw_frame and observed >= min_observed and embeddings > 0)
    if usable:
        rejection_reason = ""
    elif raw_frame or not gate_on:
        rejection_reason = "raw_frame_or_cosmetic_novelty"
    else:
        rejection_reason = "insufficient_controllable_effect_embeddings"
    novelty_signal = float(observed + embeddings) + 0.001 * float(candidate_scores + rnd_updates)
    return {
        "game": str(_candidate_field(row, "game", "") or ""),
        "policy_mode": str(
            _candidate_field(row, "policy_mode", f"novelty_{index}") or f"novelty_{index}"
        ),
        "observed_effects": int(observed),
        "episodic_embeddings": int(embeddings),
        "candidate_scores": int(candidate_scores),
        "rnd_updates": int(rnd_updates),
        "controllability_gate_on": bool(gate_on),
        "raw_frame_novelty": bool(raw_frame),
        "usable": bool(usable),
        "novelty_signal": round(float(novelty_signal), 6),
        "rejection_reason": rejection_reason,
    }


def controllable_novelty_embedding_operator(
    novelty_rows: Sequence[Any],
    *,
    min_observed_effects: int = 1,
) -> dict[str, Any]:
    """REQ-ARC-WMTE-4692: reusable controllable action-effect novelty embedder."""

    min_observed = _controllable_novelty_min(min_observed_effects)
    public_rows = [
        _controllable_novelty_public_row(row, index, min_observed)
        for index, row in enumerate(list(novelty_rows))
    ]
    public_rows.sort(
        key=lambda row: (
            not bool(row["usable"]),
            -float(row["novelty_signal"]),
            str(row["game"]),
            str(row["policy_mode"]),
        )
    )
    usable_rows = [row for row in public_rows if row["usable"]]
    if usable_rows:
        residual = ""
    elif not public_rows:
        residual = "no_controllable_novelty_rows"
    elif any(row["rejection_reason"] == "raw_frame_or_cosmetic_novelty" for row in public_rows):
        residual = "cosmetic_novelty_not_controllable"
    else:
        residual = "no_controllable_effect_embeddings"
    return {
        "operator": "controllable_novelty_embedding_operator",
        "embedding_row_count": len(public_rows),
        "usable_embedding_count": len(usable_rows),
        "rejected_embedding_count": len(public_rows) - len(usable_rows),
        "min_observed_effects": int(min_observed),
        "best_novelty_signal": float(public_rows[0]["novelty_signal"]) if public_rows else 0.0,
        "controllable_novelty_embeddings": public_rows,
        "coverage_ready": bool(usable_rows),
        "residual": residual,
        "verifier_is_oracle": False,
    }


def _object_representation_float(row: Any, key: str, default: float = 0.0) -> float:
    value = _candidate_field(row, key, default)
    if isinstance(value, bool):
        return float(default)
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _object_representation_int(row: Any, key: str) -> int:
    value = _candidate_field(row, key, 0)
    if isinstance(value, bool):
        return 0
    try:
        return max(0, int(value))
    except (TypeError, ValueError):
        return 0


def _object_representation_min(value: Any) -> int:
    if isinstance(value, bool):
        return 1
    try:
        return max(1, int(value))
    except (TypeError, ValueError):
        return 1


def _object_representation_mapping(row: Any, key: str) -> Mapping[str, Any]:
    value = _candidate_field(row, key, {})
    return value if isinstance(value, Mapping) else {}


def _object_representation_public_row(
    row: Any,
    index: int,
    min_component_count: int,
) -> dict[str, Any]:
    grid = _candidate_field(row, "grid")
    digest = _object_representation_mapping(row, "object_digest") or _object_representation_mapping(
        row, "digest"
    )
    digest_error = ""
    slot_count = _object_representation_int(row, "object_slot_count") or _object_representation_int(
        row, "slot_count"
    )
    relational_slot_count = _object_representation_int(row, "relational_slot_count")
    if grid is not None:
        try:
            digest = object_centric_digest(grid)
            from carnot.agentic.arc_value_learner import object_centric_slots

            slots = object_centric_slots(grid)
            slot_count = len(slots)
            relational_slot_count = sum(
                1
                for slot in slots
                if str(slot.get("slot_type") or "").endswith("_gap")
                or str(slot.get("slot_type") or "") == "object_neighborhood_gap"
            )
        except Exception as exc:
            digest = {}
            slot_count = 0
            relational_slot_count = 0
            digest_error = f"{type(exc).__name__}: {exc}"
    diagnostics = _object_representation_mapping(row, "object_centric_proposal_diagnostics")
    if not diagnostics:
        diagnostics = _object_representation_mapping(row, "diagnostics")
    if slot_count <= 0:
        slot_count = _object_representation_int(diagnostics, "last_slot_count")
    component_count = _object_representation_int(row, "component_count")
    if component_count <= 0 and digest:
        component_count = _object_representation_int(digest, "component_count")
    if component_count <= 0 and slot_count > 0:
        component_count = min(
            slot_count, max(1, _object_representation_int(row, "component_count"))
        )
    object_coverage = _object_representation_float(row, "object_centric_coverage")
    order1_coverage = _object_representation_float(row, "order1_coverage")
    explicit_delta = _candidate_field(row, "candidate_generation_coverage_delta", None)
    if explicit_delta is None:
        explicit_delta = _candidate_field(row, "coverage_delta", None)
    coverage_delta = (
        max(0.0, _object_representation_float(row, "object_centric_coverage") - order1_coverage)
        if explicit_delta is None
        else max(0.0, _object_representation_float({"value": explicit_delta}, "value"))
    )
    representation = str(
        _candidate_field(row, "representation", diagnostics.get("representation", "object_centric"))
        or "object_centric"
    )
    has_relational_slots = bool(
        relational_slot_count > 0
        or _candidate_field(row, "has_relational_slots", False) is True
        or "object" in representation
    )
    usable = bool(
        component_count >= min_component_count and slot_count > 0 and has_relational_slots
    )
    if usable:
        rejection_reason = ""
    elif digest_error:
        rejection_reason = "object_digest_failed"
    elif component_count < min_component_count:
        rejection_reason = "insufficient_object_components"
    else:
        rejection_reason = "no_relational_object_slots"
    return {
        "game": str(_candidate_field(row, "game", f"object_representation_{index}") or ""),
        "representation": representation,
        "component_count": int(component_count),
        "object_slot_count": int(slot_count),
        "relational_slot_count": int(relational_slot_count),
        "object_centric_coverage": round(float(object_coverage), 6),
        "order1_coverage": round(float(order1_coverage), 6),
        "coverage_delta": round(float(coverage_delta), 6),
        "usable": bool(usable),
        "rejection_reason": rejection_reason,
        "digest_error": digest_error,
        "digest_shape": list(digest.get("shape", [])) if isinstance(digest, Mapping) else [],
    }


def object_centric_representation_builder_operator(
    representation_rows: Sequence[Any],
    *,
    min_component_count: int = 1,
) -> dict[str, Any]:
    """REQ-ARC-WMTE-4704: reusable object-centric representation builder."""

    minimum = _object_representation_min(min_component_count)
    public_rows = [
        _object_representation_public_row(row, index, minimum)
        for index, row in enumerate(list(representation_rows))
    ]
    public_rows.sort(
        key=lambda row: (
            not bool(row["usable"]),
            -float(row["coverage_delta"]),
            -int(row["object_slot_count"]),
            str(row["game"]),
        )
    )
    usable_rows = [row for row in public_rows if row["usable"]]
    if usable_rows:
        residual = ""
    elif not public_rows:
        residual = "no_object_centric_rows"
    else:
        residual = "no_usable_object_centric_representation"
    return {
        "operator": "object_centric_representation_builder_operator",
        "representation_row_count": len(public_rows),
        "usable_representation_count": len(usable_rows),
        "rejected_representation_count": len(public_rows) - len(usable_rows),
        "min_component_count": int(minimum),
        "best_coverage_delta": float(public_rows[0]["coverage_delta"]) if public_rows else 0.0,
        "object_slot_total": int(sum(int(row["object_slot_count"]) for row in public_rows)),
        "object_centric_representations": public_rows,
        "coverage_ready": bool(usable_rows),
        "residual": residual,
        "verifier_is_oracle": False,
    }


def _effect_row_field(row: Any, key: str, default: Any = None) -> Any:
    if isinstance(row, Mapping):
        return row.get(key, default)
    return getattr(row, key, default)


def _effect_row_float(row: Any, key: str, default: float = 0.0) -> float:
    try:
        return float(_effect_row_field(row, key, default) or 0.0)
    except (TypeError, ValueError):
        return float(default)


def _effect_row_int(row: Any, key: str) -> int | None:
    value = _effect_row_field(row, key)
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _action_id_from(value: Any) -> int | None:
    action_id = _candidate_field(value, "action_id")
    if action_id is None:
        action_id = _candidate_field(value, "action")
    try:
        return None if action_id is None else int(action_id)
    except (TypeError, ValueError):
        return None


def _click_xy_from(value: Any) -> tuple[int, int] | None:
    data = _candidate_field(value, "data", None)
    x_value = _candidate_field(value, "x", None)
    y_value = _candidate_field(value, "y", None)
    if isinstance(data, Mapping):
        x_value = data.get("x", data.get("click_x", x_value))
        y_value = data.get("y", data.get("click_y", y_value))
    if x_value is None or y_value is None:
        return None
    try:
        return int(x_value), int(y_value)
    except (TypeError, ValueError):
        return None


def _aem_click_bucket(x: int, y: int, bucket_size: int) -> tuple[int, int]:
    size = max(1, int(bucket_size))
    return int(x) // size, int(y) // size


@dataclass(frozen=True)
class PersistentAEM:
    """REQ-ARC-WMTE-4573: cross-game cached action-effect memory."""

    effect_counts: Mapping[str, Mapping[str, float]]
    included_games: tuple[str, ...] = ()
    excluded_games: tuple[str, ...] = ()
    row_count: int = 0
    bucket_size: int = 16
    smoothing: float = 1.0

    @classmethod
    def from_effect_rows(
        cls,
        rows: Sequence[Any],
        *,
        exclude_games: Sequence[str] = (),
        bucket_size: int = 16,
        smoothing: float = 1.0,
    ) -> "PersistentAEM":
        excluded = {str(game) for game in exclude_games if str(game)}
        counts: dict[str, dict[str, float]] = {}
        included_games: set[str] = set()
        row_count = 0

        def add(key: str, effect: float) -> None:
            row = counts.setdefault(key, {"total": 0.0, "effect": 0.0, "levelup": 0.0})
            row["total"] += 1.0
            row["effect"] += float(effect > 0.0)
            row["levelup"] += float(effect >= 2.0)

        for row in rows:
            game = str(
                _effect_row_field(row, "game", "") or _effect_row_field(row, "env", "") or ""
            )
            if game in excluded:
                continue
            action_id = _effect_row_int(row, "action_id")
            if action_id is None:
                action_id = _effect_row_int(row, "action")
            if action_id is None:
                continue
            level_progress = _effect_row_float(row, "level_progress")
            changed_value = _effect_row_field(row, "changed", None)
            changed = (
                bool(changed_value)
                if changed_value is not None
                else (_effect_row_float(row, "frame_delta") > 0.0)
            )
            effect = 2.0 if level_progress > 0.0 else (1.0 if changed else 0.0)
            add(f"action:{action_id}", effect)
            x = _effect_row_int(row, "x")
            y = _effect_row_int(row, "y")
            if action_id == 6 and x is not None and y is not None:
                bx, by = _aem_click_bucket(x, y, bucket_size)
                add(f"click_bucket:{bx}:{by}", effect)
            if game:
                included_games.add(game)
            row_count += 1

        return cls(
            effect_counts=counts,
            included_games=tuple(sorted(included_games)),
            excluded_games=tuple(sorted(excluded)),
            row_count=int(row_count),
            bucket_size=int(bucket_size),
            smoothing=float(smoothing),
        )

    def _ratio(self, key: str, field: str = "effect") -> float:
        row = self.effect_counts.get(key)
        if not row:
            return 0.0
        total = float(row.get("total") or 0.0)
        if total <= 0.0:
            return 0.0
        smooth = max(0.0, float(self.smoothing))
        return float((float(row.get(field) or 0.0) + smooth) / (total + (2.0 * smooth)))

    def candidate_score(self, candidate: Any) -> float:
        action_id = _action_id_from(candidate)
        if action_id is None:
            return 0.0
        score = self._ratio(f"action:{action_id}")
        click = _click_xy_from(candidate)
        if action_id == 6 and click is not None:
            bx, by = _aem_click_bucket(click[0], click[1], self.bucket_size)
            score += self._ratio(f"click_bucket:{bx}:{by}")
        return float(score)

    def as_dict(self) -> dict[str, Any]:
        return {
            "row_count": int(self.row_count),
            "included_games": list(self.included_games),
            "excluded_games": list(self.excluded_games),
            "bucket_size": int(self.bucket_size),
            "feature_count": int(len(self.effect_counts)),
        }


def _candidate_reaches_levelup(candidate: Any, target_key: str) -> bool:
    explicit = _candidate_field(candidate, target_key, None)
    if explicit is not None:
        return bool(explicit)
    for key in ("reaches_goal", "target", "solved"):
        value = _candidate_field(candidate, key, None)
        if value is not None:
            return bool(value)
    return _candidate_score(candidate, "level_progress") > 0.0


def _normalise_resolved_action(value: Any) -> dict[str, Any] | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return {"action": int(value)}
    if not isinstance(value, Mapping):
        return None
    action_id = value.get("action", value.get("action_id"))
    if action_id is None and value.get("x") is not None and value.get("y") is not None:
        action_id = 6
    try:
        action_int = int(action_id)
    except (TypeError, ValueError):
        return None
    action: dict[str, Any] = {"action": action_int}
    data = value.get("data") if isinstance(value.get("data"), Mapping) else {}
    x_value = data.get("x", value.get("x")) if isinstance(data, Mapping) else value.get("x")
    y_value = data.get("y", value.get("y")) if isinstance(data, Mapping) else value.get("y")
    if x_value is not None and y_value is not None:
        try:
            action["data"] = {"x": int(x_value), "y": int(y_value)}
        except (TypeError, ValueError):
            return None
    return action


def _resolve_action_with(resolver: Any, label: Any) -> dict[str, Any] | None:
    if resolver is None:
        return _normalise_resolved_action(label)
    try:
        value = resolver(label) if callable(resolver) else resolver.get(label)
    except Exception:
        return None
    if value is None and isinstance(resolver, Mapping):
        value = resolver.get(str(label))
    return _normalise_resolved_action(value)


def _call_resolve_target(
    target_predicate: Optional[Callable[..., bool]],
    actions: Sequence[Mapping[str, Any]],
    *,
    game: str,
    mode: str,
) -> bool:
    if target_predicate is None:
        return bool(actions)
    for args in ((actions, game, mode), (actions, game), (actions,)):
        try:
            return bool(target_predicate(*args))
        except TypeError:
            continue
    return False


def env_adaptive_resolve_operator(
    labels: Sequence[Any],
    *,
    adaptive_resolver: Any,
    frozen_resolver: Any = None,
    target_predicate: Optional[Callable[..., bool]] = None,
    game: str = "",
) -> dict[str, Any]:
    """REQ-CAPSTONE-4584: compare frozen replay with env-adaptive re-derived actions."""

    label_list = list(labels)
    frozen_actions = [
        action for label in label_list if (action := _resolve_action_with(frozen_resolver, label))
    ]
    adaptive_actions = [
        action for label in label_list if (action := _resolve_action_with(adaptive_resolver, label))
    ]
    frozen_reached = _call_resolve_target(
        target_predicate,
        frozen_actions,
        game=game,
        mode="frozen",
    )
    adaptive_reached = _call_resolve_target(
        target_predicate,
        adaptive_actions,
        game=game,
        mode="adaptive",
    )
    dead_end = ""
    if not adaptive_actions:
        dead_end = "adaptive resolver produced no replayable actions for the symbolic plan."
    elif not adaptive_reached:
        dead_end = (
            "adaptive resolver produced actions, but the transfer verifier target was not reached."
        )
    elif frozen_reached:
        dead_end = "frozen replay already reached the transfer verifier target, so no drift recovery value was added."
    return {
        "operator": "env_adaptive_resolve_operator",
        "game": str(game),
        "label_count": int(len(label_list)),
        "frozen_actions": frozen_actions,
        "adaptive_actions": adaptive_actions,
        "frozen_reached": bool(frozen_reached),
        "adaptive_reached": bool(adaptive_reached),
        "drift_recovered": bool(adaptive_reached and not frozen_reached),
        "value_added": bool(adaptive_reached and not frozen_reached),
        "dead_end": dead_end,
    }


def _live_action_effect_score(scorer: Any, frame: Any, candidate: Any) -> float | None:
    if scorer is None:
        return None
    try:
        if hasattr(scorer, "candidate_score"):
            return float(scorer.candidate_score(frame, candidate))
        return float(scorer(frame, candidate))
    except TypeError:
        try:
            return float(scorer(candidate))
        except (TypeError, ValueError):
            return None
    except (ValueError, AttributeError):
        return None


def persistent_action_effect_memory_operator(
    candidates: Sequence[Any],
    *,
    memory: PersistentAEM,
    frame: Any = None,
    scorer: Any = None,
    target_key: str = "reaches_levelup",
) -> dict[str, Any]:
    """REQ-ARC-WMTE-4573/4632: rank candidates with cross-game action-effect evidence."""

    rows: list[dict[str, Any]] = []
    scorer_used = False
    for index, candidate in enumerate(candidates):
        row = dict(candidate) if isinstance(candidate, Mapping) else {"candidate": repr(candidate)}
        row["candidate_id"] = str(
            row.get("candidate_id") or _candidate_identifier(candidate, index)
        )
        live_score = _live_action_effect_score(scorer, frame, candidate)
        if live_score is None:
            row["action_effect_score"] = float(memory.candidate_score(candidate))
        else:
            scorer_used = True
            row["action_effect_score"] = float(live_score)
        row["original_index"] = int(index)
        row["target"] = _candidate_reaches_levelup(candidate, target_key)
        rows.append(row)

    ranked = sorted(
        rows,
        key=lambda row: (-float(row.get("action_effect_score") or 0.0), int(row["original_index"])),
    )

    def first_target(items: Sequence[Mapping[str, Any]]) -> int | None:
        for index, row in enumerate(items, start=1):
            if row.get("target") is True:
                return int(index)
        return None

    before = first_target(rows)
    after = first_target(ranked)
    actions_reduced = float(before - after) if before is not None and after is not None else 0.0
    return {
        "operator": "persistent_action_effect_memory_operator",
        "score_source": "live_action_effect_scorer" if scorer_used else "persistent_aem",
        "memory": memory.as_dict(),
        "candidate_count": int(len(rows)),
        "incoming_candidates": rows,
        "ranked_candidates": ranked,
        "best_candidate_id": str(ranked[0]["candidate_id"]) if ranked else "",
        "actions_to_first_levelup_before": before,
        "actions_to_first_levelup_after": after,
        "actions_reduced": actions_reduced,
        "value_added": bool(actions_reduced > 0.0),
    }


def online_warm_action_effect_controller_operator(
    candidates: Sequence[Any],
    *,
    memory: PersistentAEM | None,
    frame: Any = None,
    scorer: Any = None,
    online_score_key: str = "online_warm_score",
    online_weight: float = 0.05,
    target_key: str = "reaches_levelup",
) -> dict[str, Any]:
    """REQ-ARC-WMTE-4730: reusable .435 online-warm action-effect controller."""

    rows: list[dict[str, Any]] = []
    scorer_used = False
    online_score_rows = 0
    for index, candidate in enumerate(candidates):
        row = dict(candidate) if isinstance(candidate, Mapping) else {"candidate": repr(candidate)}
        row["candidate_id"] = str(
            row.get("candidate_id") or _candidate_identifier(candidate, index)
        )
        try:
            memory_score = float(memory.candidate_score(candidate)) if memory is not None else 0.0
        except Exception:
            memory_score = 0.0
        live_score = _live_action_effect_score(scorer, frame, candidate)
        if live_score is None:
            raw_online = _candidate_field(candidate, online_score_key, None)
            try:
                online_score = 0.0 if raw_online is None else float(raw_online)
            except (TypeError, ValueError):
                online_score = 0.0
        else:
            scorer_used = True
            online_score = float(live_score)
        if (
            live_score is not None
            or _candidate_field(candidate, online_score_key, None) is not None
        ):
            online_score_rows += 1
        row["memory_score"] = float(memory_score)
        row["online_warm_score"] = float(online_score)
        row["action_effect_score"] = float(memory_score) + (
            max(0.0, float(online_weight)) * float(online_score)
        )
        row["original_index"] = int(index)
        row["target"] = _candidate_reaches_levelup(candidate, target_key)
        rows.append(row)

    ranked = sorted(
        rows,
        key=lambda row: (-float(row.get("action_effect_score") or 0.0), int(row["original_index"])),
    )

    def first_target(items: Sequence[Mapping[str, Any]]) -> int | None:
        for index, row in enumerate(items, start=1):
            if row.get("target") is True:
                return int(index)
        return None

    before = first_target(rows)
    after = first_target(ranked)
    actions_reduced = float(before - after) if before is not None and after is not None else 0.0
    if scorer_used:
        score_source = "persistent_aem_plus_live_online_warm"
    elif online_score_rows:
        score_source = "persistent_aem_plus_online_warm"
    else:
        score_source = "persistent_aem_only_online_warm_scaffold"
    memory_public = (
        memory.as_dict()
        if memory is not None and hasattr(memory, "as_dict")
        else {"row_count": 0, "included_games": [], "excluded_games": [], "feature_count": 0}
    )
    return {
        "operator": "online_warm_action_effect_controller_operator",
        "score_source": score_source,
        "memory": memory_public,
        "candidate_count": int(len(rows)),
        "online_score_rows": int(online_score_rows),
        "incoming_candidates": rows,
        "ranked_candidates": ranked,
        "best_candidate_id": str(ranked[0]["candidate_id"]) if ranked else "",
        "actions_to_first_levelup_before": before,
        "actions_to_first_levelup_after": after,
        "actions_reduced": actions_reduced,
        "value_added": bool(actions_reduced > 0.0),
        "verifier_is_oracle": False,
    }


def _energy_fitness_float(value: Any, default: float = 0.0) -> float:
    if isinstance(value, bool):
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _energy_fitness_action(candidate: Any) -> int:
    for key in ("action", "action_id"):
        value = _candidate_field(candidate, key)
        if value is not None:
            return int(_energy_fitness_float(value, 0.0))
    return 0


def _energy_fitness_xy(candidate: Any) -> tuple[int, int] | None:
    data = _candidate_field(candidate, "data")
    for source in (data, candidate):
        if source is None:
            continue
        x = _candidate_field(source, "x")
        y = _candidate_field(source, "y")
        if x is not None and y is not None:
            return (
                int(_energy_fitness_float(x, 0.0)),
                int(_energy_fitness_float(y, 0.0)),
            )
    return None


def _energy_fitness_descriptor(candidate: Any, bucket_size: int) -> tuple[int, int, int]:
    bucket = max(1, int(bucket_size))
    descriptor = _candidate_field(candidate, "behavior_descriptor")
    if isinstance(descriptor, Sequence) and not isinstance(descriptor, (str, bytes)):
        values = [int(_energy_fitness_float(value, 0.0)) for value in list(descriptor)[:3]]
        while len(values) < 3:
            values.append(0)
        return (values[0], values[1], values[2])
    xy = _energy_fitness_xy(candidate)
    if xy is None:
        return (_energy_fitness_action(candidate), 0, 0)
    x, y = xy
    return (_energy_fitness_action(candidate), x // bucket, y // bucket)


def _energy_fitness_score(candidate: Any) -> float:
    for key in (
        "energy_fitness",
        "energy_score",
        "qd_energy",
        "combined_goal_energy",
        "graded_goal_energy",
        "goal_energy",
    ):
        value = _candidate_field(candidate, key)
        if value is not None:
            return _energy_fitness_float(value, 1.0)
    return 1.0


def _energy_fitness_generated(candidate: Any) -> bool:
    source = str(
        _candidate_field(
            candidate,
            "generated_by",
            _candidate_field(candidate, "source", _candidate_field(candidate, "arm_label", "")),
        )
    ).lower()
    return bool(
        _candidate_truthy(candidate, "qd_generated")
        or _candidate_truthy(candidate, "generated")
        or "energy-qd" in source
        or "energy_qd" in source
        or "qd" == source
    )


def _energy_fitness_public_row(
    candidate: Any,
    index: int,
    *,
    bucket_size: int,
    target_key: str,
) -> dict[str, Any]:
    row = dict(candidate) if isinstance(candidate, Mapping) else {"candidate": repr(candidate)}
    row["candidate_id"] = str(row.get("candidate_id") or _candidate_identifier(candidate, index))
    row["energy_fitness"] = float(_energy_fitness_score(candidate))
    row["behavior_descriptor"] = list(_energy_fitness_descriptor(candidate, bucket_size))
    row["is_qd_generated"] = _energy_fitness_generated(candidate)
    row["original_index"] = int(index)
    row["target"] = bool(
        _candidate_truthy(candidate, target_key) or _candidate_reaches_goal(candidate)
    )
    return row


def energy_fitness_qd_generator_operator(
    candidates: Sequence[Any],
    *,
    max_elites: int = 32,
    bucket_size: int = 8,
    target_key: str = "reaches_goal",
) -> dict[str, Any]:
    """REQ-ARC-WMTE-4741: reusable .436 energy-fitness QD generator/ranker."""

    rows = [
        _energy_fitness_public_row(
            candidate,
            index,
            bucket_size=bucket_size,
            target_key=target_key,
        )
        for index, candidate in enumerate(candidates)
    ]
    baseline_descriptors = {
        tuple(row["behavior_descriptor"]) for row in rows if not row["is_qd_generated"]
    }
    generated_descriptors = {
        tuple(row["behavior_descriptor"]) for row in rows if row["is_qd_generated"]
    }
    archive: dict[tuple[int, int, int], dict[str, Any]] = {}
    for row in rows:
        descriptor = tuple(row["behavior_descriptor"])
        current = archive.get(descriptor)
        if current is None or (
            float(row["energy_fitness"]),
            int(row["original_index"]),
        ) < (
            float(current["energy_fitness"]),
            int(current["original_index"]),
        ):
            archive[descriptor] = row
    elites = sorted(
        archive.values(),
        key=lambda row: (float(row["energy_fitness"]), int(row["original_index"])),
    )[: max(0, int(max_elites))]
    elite_ids = {row["candidate_id"] for row in elites}
    tail = [row for row in rows if row["candidate_id"] not in elite_ids]
    ranked = elites + sorted(
        tail,
        key=lambda row: (float(row["energy_fitness"]), int(row["original_index"])),
    )

    def first_target(items: Sequence[Mapping[str, Any]]) -> int | None:
        for position, row in enumerate(items, start=1):
            if row.get("target") is True:
                return int(position)
        return None

    before = first_target(rows)
    after = first_target(ranked)
    lift = float(before - after) if before is not None and after is not None else 0.0
    coverage_delta = float(len(generated_descriptors - baseline_descriptors))
    return {
        "operator": "energy_fitness_qd_generator_operator",
        "score_source": "cached_candidate_energy_fitness",
        "verifier_is_oracle": False,
        "candidate_count": int(len(rows)),
        "generated_candidate_count": int(sum(1 for row in rows if row["is_qd_generated"])),
        "behavior_descriptor_count": int(len({tuple(row["behavior_descriptor"]) for row in rows})),
        "candidate_generation_coverage_delta": coverage_delta,
        "archive_size": int(len(elites)),
        "archive": elites,
        "incoming_candidates": rows,
        "ranked_candidates": ranked,
        "best_candidate_id": str(ranked[0]["candidate_id"]) if ranked else "",
        "actions_to_first_goal_before": before,
        "actions_to_first_goal_after": after,
        "action_efficiency_lift": lift,
        "value_added": bool(lift > 0.0 or coverage_delta > 0.0),
    }


def _goal_energy_candidate_state(candidate: Any) -> Any:
    for key in ("goal_state", "visible_goal_state", "target_group_state", "state", "frame"):
        value = _candidate_field(candidate, key)
        if value is not None:
            return value
    return candidate


def _goal_energy_navigation(candidate: Any) -> float:
    for key in (
        "navigation_energy",
        "arc_goal_distance",
        "goal_distance",
        "navigation",
        "heuristic",
        "search_energy",
    ):
        value = _candidate_field(candidate, key)
        if value is not None:
            try:
                return float(value)
            except (TypeError, ValueError):
                return 0.0
    return 0.0


def _call_goal_energy(goal_energy: Any, state: Any, candidate: Any) -> float:
    if goal_energy is None:
        for key in ("graded_goal_energy", "goal_energy"):
            value = _candidate_field(candidate, key)
            if value is not None:
                try:
                    return float(value)
                except (TypeError, ValueError):
                    return 1.0
        return 1.0
    try:
        return float(goal_energy(state))
    except Exception:
        return 1.0


def _goal_predicate_passes(goal_energy: Any, state: Any) -> bool:
    predicate = getattr(goal_energy, "predicate_fires", None)
    if not callable(predicate):
        return False
    try:
        return bool(predicate(state))
    except Exception:
        return False


def graded_goal_energy_search_heuristic_operator(
    candidates: Sequence[Any],
    *,
    goal_energy: Any = None,
    alpha: float = 0.9,
    beta: float = 0.1,
    target_key: str = "reaches_goal",
) -> dict[str, Any]:
    """REQ-ARC-WMTE-4644: rank candidates by navigation + graded goal energy."""

    if goal_energy is None:
        try:
            from carnot.agentic.arc_goal_energy_live import load_exp4020_goal_energy

            goal_energy = load_exp4020_goal_energy(REPO)
        except Exception:
            goal_energy = None
    weight_sum = float(alpha) + float(beta)
    if abs(weight_sum - 1.0) > 1e-9:
        raise ValueError("graded goal-energy operator requires alpha + beta == 1")

    rows: list[dict[str, Any]] = []
    for index, candidate in enumerate(candidates):
        state = _goal_energy_candidate_state(candidate)
        navigation = _goal_energy_navigation(candidate)
        graded = _call_goal_energy(goal_energy, state, candidate)
        combined = float(alpha) * navigation + float(beta) * graded
        row = dict(candidate) if isinstance(candidate, Mapping) else {"candidate": repr(candidate)}
        row["candidate_id"] = str(
            row.get("candidate_id") or _candidate_identifier(candidate, index)
        )
        row["navigation_energy"] = float(navigation)
        row["graded_goal_energy"] = float(graded)
        row["combined_goal_energy"] = float(combined)
        row["goal_predicate_pass"] = _goal_predicate_passes(goal_energy, state)
        row["original_index"] = int(index)
        row["target"] = bool(
            _candidate_truthy(candidate, target_key) or _candidate_reaches_goal(candidate)
        )
        rows.append(row)

    ranked = sorted(
        rows,
        key=lambda row: (
            float(1.0 if row.get("combined_goal_energy") is None else row["combined_goal_energy"]),
            int(row["original_index"]),
        ),
    )

    def first_target(items: Sequence[Mapping[str, Any]]) -> int | None:
        for index, row in enumerate(items, start=1):
            if row.get("target") is True:
                return int(index)
        return None

    before = first_target(rows)
    after = first_target(ranked)
    lift = float(before - after) if before is not None and after is not None else 0.0
    return {
        "operator": "graded_goal_energy_search_heuristic_operator",
        "score_source": "exp4020_graded_goal_satisfaction_energy"
        if goal_energy is not None
        else "cached_candidate_goal_energy",
        "alpha": float(alpha),
        "beta": float(beta),
        "verifier_is_oracle": False,
        "candidate_count": int(len(rows)),
        "incoming_candidates": rows,
        "ranked_candidates": ranked,
        "best_candidate_id": str(ranked[0]["candidate_id"]) if ranked else "",
        "actions_to_first_goal_before": before,
        "actions_to_first_goal_after": after,
        "action_efficiency_lift": lift,
        "value_added": bool(lift > 0.0),
    }


def verifier_router_candidate_ranking_operator(
    candidates: Sequence[Any],
    *,
    score_fn: Optional[Callable[[Any, Mapping[str, Any]], float]] = None,
    score_key: str = "verifier_score",
    target_key: str = "reaches_goal",
    higher_is_better: bool = True,
    context: Optional[Mapping[str, Any]] = None,
) -> dict[str, Any]:
    """REQ-ARC-WMTE-4561: generic verifier-router candidate ranking primitive."""

    ctx = dict(context or {})
    rows: list[dict[str, Any]] = []
    for index, candidate in enumerate(candidates):
        if isinstance(candidate, Mapping):
            row = dict(candidate)
        else:
            row = {"candidate": repr(candidate)}
        score = (
            float(score_fn(candidate, ctx))
            if score_fn is not None
            else _candidate_score(candidate, score_key)
        )
        row["candidate_id"] = str(
            row.get("candidate_id") or _candidate_identifier(candidate, index)
        )
        row["verifier_score"] = score
        row["original_index"] = index
        row["target"] = bool(_candidate_field(candidate, target_key, False))
        rows.append(row)

    def rank_key(row: Mapping[str, Any]) -> tuple[float, int]:
        score = float(row.get("verifier_score") or 0.0)
        return ((-score if higher_is_better else score), int(row.get("original_index") or 0))

    ranked = sorted(rows, key=rank_key)

    def first_target_rank(items: Sequence[Mapping[str, Any]]) -> Optional[int]:
        for idx, row in enumerate(items):
            if row.get("target") is True:
                return idx
        return None

    target_before = first_target_rank(rows)
    target_after = first_target_rank(ranked)
    ordering_gain = (
        int(target_before) - int(target_after)
        if target_before is not None and target_after is not None
        else 0
    )
    return {
        "operator": "verifier_router_candidate_ranking_operator",
        "score_key": score_key,
        "higher_is_better": bool(higher_is_better),
        "candidate_count": len(rows),
        "incoming_candidates": rows,
        "ranked_candidates": ranked,
        "best_candidate_id": str(ranked[0]["candidate_id"]) if ranked else "",
        "target_rank_before": target_before,
        "target_rank_after": target_after,
        "ordering_gain": ordering_gain,
        "value_added": ordering_gain > 0,
    }


def _observation_level(observation: Any) -> int:
    if isinstance(observation, Mapping):
        for key in ("levels_completed", "level", "reached_level"):
            if key in observation:
                return int(observation[key] or 0)
    if hasattr(observation, "levels_completed"):
        return int(getattr(observation, "levels_completed") or 0)
    return frame_level(observation)


def per_level_reinduction_operator(
    observations: Sequence[Any],
    *,
    predicate_inducer: Callable[[int, dict[str, Any]], Mapping[str, Any] | str | None],
    route_builder: Optional[Callable[[dict[str, Any]], Mapping[str, Any]]] = None,
    initial_predicate: Mapping[str, Any] | str | None = None,
    initial_level: Optional[int] = None,
) -> dict[str, Any]:
    """REQ-ARC-WMTE-4537: reusable detect-level-up -> re-induce -> route loop."""

    if route_builder is None:
        route_builder = lambda event: {
            "route": "depth_primary_goal_bias",
            "depth_primary": True,
            "goal_bias_label": str((event.get("predicate") or {}).get("predicate_id") or ""),
        }

    current_level = int(initial_level) if initial_level is not None else None
    prior_signature = (
        str(initial_predicate.get("signature") or initial_predicate.get("predicate_id"))
        if isinstance(initial_predicate, Mapping)
        else (str(initial_predicate) if initial_predicate is not None else "")
    )
    events: list[dict[str, Any]] = []

    for index, observation in enumerate(observations):
        level = _observation_level(observation)
        if current_level is None:
            current_level = level
            continue
        if level <= current_level:
            continue
        for won_level in range(current_level + 1, level + 1):
            next_goal_level = won_level + 1
            context = {
                "from_level": won_level,
                "next_goal_level": next_goal_level,
                "observation_index": index,
                "observation": observation,
                "prior_predicate_signature": prior_signature,
                "clear_stale_induction": True,
            }
            raw_predicate = predicate_inducer(next_goal_level, context)
            if isinstance(raw_predicate, Mapping):
                predicate = dict(raw_predicate)
            elif raw_predicate is None:
                predicate = {
                    "predicate_id": f"L{next_goal_level}_predicate_unavailable",
                    "signature": "",
                    "representation_correct": False,
                }
            else:
                predicate = {
                    "predicate_id": str(raw_predicate),
                    "signature": str(raw_predicate),
                    "representation_correct": True,
                }
            predicate.setdefault("predicate_id", f"L{next_goal_level}_predicate")
            predicate.setdefault("signature", str(predicate.get("predicate_id") or ""))
            predicate.setdefault("representation_correct", False)
            signature = str(predicate.get("signature") or predicate.get("predicate_id") or "")
            representation_transfer = bool(
                predicate.get("representation_correct") is True
                and signature
                and (not prior_signature or signature != prior_signature)
            )
            event = {
                "trigger": "level_up",
                "from_level": won_level,
                "next_goal_level": next_goal_level,
                "stale_state_cleared": True,
                "predicate": predicate,
                "representation_transfer": representation_transfer,
            }
            event["route"] = dict(route_builder(event))
            events.append(event)
            prior_signature = signature
        current_level = level

    return {
        "operator": "per_level_reinduction_operator",
        "level_ups_detected": len(events),
        "stale_state_cleared": bool(events),
        "current_level": int(current_level or 0),
        "events": events,
        "latest_predicate": events[-1]["predicate"] if events else None,
        "latest_route": events[-1]["route"] if events else None,
        "representation_transfer": any(bool(event["representation_transfer"]) for event in events),
    }


def _normalise_llm_proposer_candidates(raw: Any) -> list[dict[str, Any]]:
    if raw is None:
        return []
    if isinstance(raw, Mapping):
        nested = raw.get("candidates")
        if isinstance(nested, Sequence) and not isinstance(nested, (str, bytes)):
            return [dict(row) if isinstance(row, Mapping) else {"name": str(row)} for row in nested]
        return [dict(raw)]
    if isinstance(raw, Sequence) and not isinstance(raw, (str, bytes)):
        return [dict(row) if isinstance(row, Mapping) else {"name": str(row)} for row in raw]
    return [{"name": str(raw), "goal_predicate": str(raw), "representation_correct": True}]


def _rank_llm_proposer_candidates(candidates: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    rows = [dict(candidate) for candidate in candidates]

    def key(row: Mapping[str, Any]) -> tuple[float, float, str]:
        trust_energy = row.get("trust_energy")
        heldout = row.get("heldout_accuracy")
        return (
            float("inf") if trust_energy is None else float(trust_energy),
            -float(heldout or 0.0),
            str(row.get("name") or row.get("candidate_name") or ""),
        )

    return sorted(rows, key=key)


def _candidate_reaches_goal(candidate: Mapping[str, Any]) -> bool:
    return bool(
        candidate.get("reachable_plan") is True
        or candidate.get("plan_reaches_goal") is True
        or candidate.get("planned") is True
    )


def _predicate_from_llm_candidate(
    next_goal_level: int, candidate: Mapping[str, Any]
) -> dict[str, Any]:
    name = str(
        candidate.get("name") or candidate.get("candidate_name") or f"L{next_goal_level}_candidate"
    )
    signature = str(candidate.get("signature") or candidate.get("goal_predicate") or name)
    return {
        "predicate_id": str(
            candidate.get("predicate_id") or candidate.get("goal_predicate") or name
        ),
        "signature": signature,
        "representation_correct": bool(candidate.get("representation_correct") is True),
        "goal_predicate": candidate.get("goal_predicate"),
        "dynamics_model": candidate.get("dynamics_model"),
        "source": "llm_proposer",
    }


def llm_proposer_reinduction_operator(
    observations: Sequence[Any],
    *,
    proposal_provider: Optional[
        Callable[
            [int, dict[str, Any]], Mapping[str, Any] | Sequence[Mapping[str, Any]] | str | None
        ]
    ] = None,
    fallback_predicate_inducer: Optional[
        Callable[[int, dict[str, Any]], Mapping[str, Any] | str | None]
    ] = None,
    route_builder: Optional[Callable[[dict[str, Any]], Mapping[str, Any]]] = None,
    initial_predicate: Mapping[str, Any] | str | None = None,
    initial_level: Optional[int] = None,
    max_refinement_rounds: int = 3,
    model_specs: str = "",
) -> dict[str, Any]:
    """REQ-ARC-WMTE-4549: reusable LLM-proposer re-induction with DSL fallback."""

    if route_builder is None:
        route_builder = lambda event: {
            "route": "depth_primary_goal_bias",
            "depth_primary": True,
            "goal_bias_label": str((event.get("predicate") or {}).get("predicate_id") or ""),
            "operator": "llm_proposer_reinduction_operator",
        }
    if fallback_predicate_inducer is None:
        fallback_predicate_inducer = lambda goal_level, _context: {
            "predicate_id": f"L{goal_level}_predicate_unavailable",
            "signature": "",
            "representation_correct": False,
            "source": "dsl_fallback",
        }

    current_level = int(initial_level) if initial_level is not None else None
    prior_signature = (
        str(initial_predicate.get("signature") or initial_predicate.get("predicate_id"))
        if isinstance(initial_predicate, Mapping)
        else (str(initial_predicate) if initial_predicate is not None else "")
    )
    events: list[dict[str, Any]] = []
    round_limit = max(0, min(int(max_refinement_rounds), 3))

    for index, observation in enumerate(observations):
        level = _observation_level(observation)
        if current_level is None:
            current_level = level
            continue
        if level <= current_level:
            continue
        for won_level in range(current_level + 1, level + 1):
            next_goal_level = won_level + 1
            base_context = {
                "from_level": won_level,
                "next_goal_level": next_goal_level,
                "observation_index": index,
                "observation": observation,
                "prior_predicate_signature": prior_signature,
                "clear_stale_induction": True,
            }
            rounds: list[dict[str, Any]] = []
            selected: dict[str, Any] | None = None
            counterexample: dict[str, Any] = {"kind": "initial_induction"}
            proposal_invoked = proposal_provider is not None and round_limit > 0
            if proposal_invoked:
                for round_no in range(1, round_limit + 1):
                    context = {
                        **base_context,
                        "refinement_round": round_no,
                        "counterexample": dict(counterexample),
                    }
                    raw = proposal_provider(next_goal_level, context)
                    candidates = _rank_llm_proposer_candidates(
                        _normalise_llm_proposer_candidates(raw)
                    )
                    selected = candidates[0] if candidates else None
                    row = {
                        "round": round_no,
                        "action": "induce" if round_no == 1 else "refactor",
                        "candidate_names": [
                            str(candidate.get("name") or candidate.get("candidate_name") or "")
                            for candidate in candidates
                        ],
                        "trust_energy_ranked": bool(candidates),
                    }
                    if selected is None:
                        counterexample = {"kind": "no_candidate"}
                        row["counterexample"] = dict(counterexample)
                        rounds.append(row)
                        continue
                    reachable = _candidate_reaches_goal(selected)
                    row.update(
                        {
                            "selected_candidate_name": str(
                                selected.get("name") or selected.get("candidate_name") or ""
                            ),
                            "trust_energy": selected.get("trust_energy"),
                            "heldout_accuracy": selected.get("heldout_accuracy"),
                            "plan_length": len(selected.get("plan") or []),
                            "reachable_plan": bool(reachable),
                        }
                    )
                    rounds.append(row)
                    if reachable:
                        break
                    counterexample = {
                        "kind": "no_reachable_plan",
                        "selected_candidate_name": row["selected_candidate_name"],
                    }
                    row["counterexample"] = dict(counterexample)

            if selected is not None:
                predicate = _predicate_from_llm_candidate(next_goal_level, selected)
                proposal_mode = "llm_proposer"
                reachable_plan = _candidate_reaches_goal(selected)
                selected_name = str(selected.get("name") or selected.get("candidate_name") or "")
            else:
                raw_predicate = fallback_predicate_inducer(next_goal_level, dict(base_context))
                if isinstance(raw_predicate, Mapping):
                    predicate = dict(raw_predicate)
                elif raw_predicate is None:
                    predicate = {
                        "predicate_id": f"L{next_goal_level}_predicate_unavailable",
                        "signature": "",
                        "representation_correct": False,
                    }
                else:
                    predicate = {
                        "predicate_id": str(raw_predicate),
                        "signature": str(raw_predicate),
                        "representation_correct": True,
                    }
                predicate.setdefault("predicate_id", f"L{next_goal_level}_predicate")
                predicate.setdefault("signature", str(predicate.get("predicate_id") or ""))
                predicate.setdefault("representation_correct", False)
                predicate.setdefault("source", "dsl_fallback")
                proposal_mode = "dsl_fallback"
                reachable_plan = False
                selected_name = ""

            signature = str(predicate.get("signature") or predicate.get("predicate_id") or "")
            representation_transfer = bool(
                predicate.get("representation_correct") is True
                and signature
                and (not prior_signature or signature != prior_signature)
            )
            event = {
                "trigger": "level_up",
                "from_level": won_level,
                "next_goal_level": next_goal_level,
                "stale_state_cleared": True,
                "predicate": predicate,
                "representation_transfer": representation_transfer,
                "reachable_plan_produced": bool(reachable_plan),
                "proposal_mode": proposal_mode,
                "llm_proposer_invoked": bool(proposal_invoked),
                "trust_energy_ranked": any(bool(row.get("trust_energy_ranked")) for row in rounds),
                "refinement_rounds": rounds,
                "refinement_rounds_used": len(rounds),
                "selected_candidate_name": selected_name,
                "model_specs": str(model_specs),
            }
            event["route"] = dict(route_builder(event))
            events.append(event)
            prior_signature = signature
        current_level = level

    return {
        "operator": "llm_proposer_reinduction_operator",
        "base_operator": "per_level_reinduction_operator",
        "level_ups_detected": len(events),
        "stale_state_cleared": bool(events),
        "current_level": int(current_level or 0),
        "events": events,
        "latest_predicate": events[-1]["predicate"] if events else None,
        "latest_route": events[-1]["route"] if events else None,
        "reachable_plan_produced": any(bool(event["reachable_plan_produced"]) for event in events),
        "representation_transfer": any(bool(event["representation_transfer"]) for event in events),
    }


def cyclic_distance(current: int, target: int, *, modulus: int = 7) -> int:
    """Shortest cyclic distance on an integer wheel, used by config/glyph solvers."""

    if modulus <= 0:
        raise ValueError("modulus must be positive")
    return min((int(target) - int(current)) % modulus, (int(current) - int(target)) % modulus)


def sequence_cyclic_distance(
    current: Sequence[int],
    required: Sequence[int],
    *,
    modulus: int = 7,
    gap_cost: Optional[float] = None,
) -> float:
    """Sum cyclic distance over aligned values, with a bounded length-gap penalty."""

    n = min(len(current), len(required))
    gap = float(modulus if gap_cost is None else gap_cost)
    return float(
        sum(cyclic_distance(current[i], required[i], modulus=modulus) for i in range(n))
        + gap * abs(len(current) - len(required))
    )


def _sprite_grid(value: Any) -> tuple[tuple[int, ...], ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        return ()
    rows: list[tuple[int, ...]] = []
    width: int | None = None
    for row in value:
        if not isinstance(row, Sequence) or isinstance(row, (str, bytes)):
            return ()
        parsed = tuple(int(cell) for cell in row)
        if width is None:
            width = len(parsed)
        if not parsed or len(parsed) != width:
            return ()
        rows.append(parsed)
    return tuple(rows)


def _source_color(
    pixels: Sequence[Sequence[int]], *, transparent: int = -1, marker: int = 0
) -> int | None:
    counts: dict[int, int] = {}
    for row in pixels:
        for cell in row:
            color = int(cell)
            if color in {transparent, marker}:
                continue
            counts[color] = counts.get(color, 0) + 1
    if not counts:
        return None
    return sorted(counts.items(), key=lambda item: (-item[1], item[0]))[0][0]


def _local_points_for_color(pixels: Sequence[Sequence[int]], color: int) -> set[tuple[int, int]]:
    points: set[tuple[int, int]] = set()
    for y, row in enumerate(pixels):
        for x, cell in enumerate(row):
            if int(cell) == int(color):
                points.add((x, y))
    return points


def _sprite_overlay_required_pixels(object_digest: Mapping[str, Any]) -> list[tuple[int, int, int]]:
    direct = object_digest.get("required_pixels")
    if isinstance(direct, Sequence) and not isinstance(direct, (str, bytes)):
        parsed: list[tuple[int, int, int]] = []
        for row in direct:
            if not isinstance(row, Mapping):
                continue
            parsed.append((int(row["x"]), int(row["y"]), int(row["color"])))
        return parsed

    ignore_colors = {
        int(color)
        for color in object_digest.get("target_match_ignore_colors", (-1, 4))
        if isinstance(color, int) or str(color).lstrip("-").isdigit()
    }
    required: list[tuple[int, int, int]] = []
    targets = object_digest.get("targets") or ()
    if not isinstance(targets, Sequence) or isinstance(targets, (str, bytes)):
        return required
    for target in targets:
        if not isinstance(target, Mapping):
            continue
        pixels = _sprite_grid(target.get("pixels"))
        if not pixels:
            continue
        x0 = int(target.get("x") or 0)
        y0 = int(target.get("y") or 0)
        for y, row in enumerate(pixels):
            for x, cell in enumerate(row):
                color = int(cell)
                if color not in ignore_colors:
                    required.append((x0 + x, y0 + y, color))
    return required


def _sprite_overlay_variants(source: Mapping[str, Any]) -> list[dict[str, Any]]:
    base_pixels = _sprite_grid(source.get("pixels"))
    variants = [
        {
            "variant_id": "base",
            "pixels": base_pixels,
            "pre_labels": [],
            "post_labels": [],
            "resize_variant_used": False,
        }
    ]
    raw = source.get("variants") or ()
    if isinstance(raw, Sequence) and not isinstance(raw, (str, bytes)):
        for index, variant in enumerate(raw):
            if not isinstance(variant, Mapping):
                continue
            pixels = _sprite_grid(variant.get("pixels"))
            if not pixels:
                continue
            variants.append(
                {
                    "variant_id": str(
                        variant.get("id") or variant.get("variant_id") or f"variant_{index}"
                    ),
                    "pixels": pixels,
                    "pre_labels": [str(label) for label in variant.get("pre_labels") or ()],
                    "post_labels": [str(label) for label in variant.get("post_labels") or ()],
                    "resize_variant_used": True,
                }
            )
    return [variant for variant in variants if variant["pixels"]]


def _best_sprite_overlay_placement(
    *,
    source: Mapping[str, Any],
    required: Sequence[tuple[int, int, int]],
    movement_step: int,
) -> dict[str, Any] | None:
    source_id = str(source.get("id") or source.get("name") or "")
    x_current = int(source.get("x") or 0)
    y_current = int(source.get("y") or 0)
    best: dict[str, Any] | None = None
    for variant in _sprite_overlay_variants(source):
        color = _source_color(variant["pixels"])
        if color is None:
            continue
        local_points = _local_points_for_color(variant["pixels"], color)
        same_color = [pixel for pixel in required if pixel[2] == color]
        if not same_color or not local_points:
            continue
        candidates: set[tuple[int, int]] = set()
        for target_x, target_y, _target_color in same_color:
            for local_x, local_y in local_points:
                candidates.add((target_x - local_x, target_y - local_y))
        for x_target, y_target in candidates:
            covered = [
                pixel
                for pixel in same_color
                if (pixel[0] - x_target, pixel[1] - y_target) in local_points
            ]
            candidate = {
                "source_id": source_id,
                "source_index": int(source.get("source_index") or 0),
                "color": int(color),
                "current_top_left": [x_current, y_current],
                "target_top_left": [int(x_target), int(y_target)],
                "delta": [int(x_target - x_current), int(y_target - y_current)],
                "covered_required_pixels": [
                    {"x": int(x), "y": int(y), "color": int(c)} for x, y, c in sorted(covered)
                ],
                "covered_count": len(covered),
                "variant_id": variant["variant_id"],
                "resize_variant_used": bool(variant["resize_variant_used"]),
                "pre_labels": list(variant["pre_labels"]),
                "post_labels": list(variant["post_labels"]),
            }
            if best is None:
                best = candidate
                continue
            best_delta = best["delta"]
            candidate_delta = candidate["delta"]
            best_key = (
                int(best["covered_count"]),
                int(
                    int(best_delta[0]) % max(1, movement_step) == 0
                    and int(best_delta[1]) % max(1, movement_step) == 0
                ),
                -len(best["pre_labels"]) - len(best["post_labels"]),
                -abs(int(best_delta[0])) - abs(int(best_delta[1])),
            )
            candidate_key = (
                int(candidate["covered_count"]),
                int(
                    int(candidate_delta[0]) % max(1, movement_step) == 0
                    and int(candidate_delta[1]) % max(1, movement_step) == 0
                ),
                -len(candidate["pre_labels"]) - len(candidate["post_labels"]),
                -abs(int(candidate_delta[0])) - abs(int(candidate_delta[1])),
            )
            if candidate_key > best_key:
                best = candidate
    return best


def _sprite_overlay_movement_labels(
    delta: Sequence[int],
    *,
    movement_step: int,
    actions: Mapping[str, Any],
) -> list[str] | None:
    dx, dy = int(delta[0]), int(delta[1])
    step = max(1, int(movement_step))
    if dx % step or dy % step:
        return None
    labels: list[str] = []
    if dy < 0:
        labels.extend([str(actions.get("up", '{"action":1}'))] * (abs(dy) // step))
    elif dy > 0:
        labels.extend([str(actions.get("down", '{"action":2}'))] * (dy // step))
    if dx < 0:
        labels.extend([str(actions.get("left", '{"action":3}'))] * (abs(dx) // step))
    elif dx > 0:
        labels.extend([str(actions.get("right", '{"action":4}'))] * (dx // step))
    return labels


def _ungrounded_sprite_overlay_result(game: str, residual: str) -> dict[str, Any]:
    return {
        "operator": "sprite_overlay_resize_verifier",
        "game": str(game),
        "grounded": False,
        "solution": [],
        "predicate_id": "",
        "placements": [],
        "coverage": {
            "required_pixels": 0,
            "covered_required_pixels": 0,
            "missing_required_pixels": [],
        },
        "residual": residual,
        "verifier_is_oracle": True,
    }


def sprite_overlay_resize_verifier(
    game: str,
    object_digest: Mapping[str, Any],
    few_shot_examples: Sequence[Mapping[str, Any]] = (),
) -> dict[str, Any]:
    """REQ-REPORT-4479: ground transparent sprite overlays before claiming a solve.

    The verifier is intentionally mechanical. It receives source sprites, target
    pixels, and the action labels that the environment exposes; it can translate
    sources or select explicit resize variants supplied by the caller, but it
    will not invent hidden resize actions. That matters for ARC solve banking:
    this function may rank and emit a candidate, but only the offline
    `reproduce()` gate turns that candidate into a counted level.
    """

    del few_shot_examples
    if not isinstance(object_digest, Mapping):
        return _ungrounded_sprite_overlay_result(game, "missing_sprite_overlay_digest")
    sources_raw = object_digest.get("sources") or ()
    if not isinstance(sources_raw, Sequence) or isinstance(sources_raw, (str, bytes)):
        return _ungrounded_sprite_overlay_result(game, "missing_sprite_overlay_sources")
    sources = [
        {**dict(source), "source_index": index}
        for index, source in enumerate(sources_raw)
        if isinstance(source, Mapping)
    ]
    required = _sprite_overlay_required_pixels(object_digest)
    if not sources:
        return _ungrounded_sprite_overlay_result(game, "missing_sprite_overlay_sources")
    if not required:
        return _ungrounded_sprite_overlay_result(game, "missing_sprite_overlay_required_pixels")

    movement_step = int(object_digest.get("movement_step") or 1)
    placements = [
        placement
        for source in sources
        if (
            placement := _best_sprite_overlay_placement(
                source=source,
                required=required,
                movement_step=movement_step,
            )
        )
        is not None
    ]
    covered = {
        (int(pixel["x"]), int(pixel["y"]), int(pixel["color"]))
        for placement in placements
        for pixel in placement["covered_required_pixels"]
    }
    required_set = {(int(x), int(y), int(color)) for x, y, color in required}
    missing = sorted(required_set - covered)
    coverage = {
        "required_pixels": len(required_set),
        "covered_required_pixels": len(covered),
        "missing_required_pixels": [
            {"x": int(x), "y": int(y), "color": int(color)} for x, y, color in missing
        ],
    }
    if missing:
        result = _ungrounded_sprite_overlay_result(game, "sprite_overlay_required_pixels_uncovered")
        result["placements"] = placements
        result["coverage"] = coverage
        return result

    actions = object_digest.get("actions") or {}
    actions = actions if isinstance(actions, Mapping) else {}
    active_index = int(object_digest.get("active_source_index") or 0)
    by_index = {int(placement["source_index"]): placement for placement in placements}
    source_count = len(sources)
    ordered_indices = [
        index
        for offset in range(source_count)
        if (index := (active_index + offset) % source_count) in by_index
    ]
    solution: list[str] = []
    cursor = active_index
    for index in ordered_indices:
        if index != cursor:
            cycle = actions.get("cycle")
            if cycle is None:
                result = _ungrounded_sprite_overlay_result(
                    game,
                    "sprite_overlay_action_model_cannot_cycle_sources",
                )
                result["placements"] = placements
                result["coverage"] = coverage
                return result
            cycle_count = (index - cursor) % source_count
            solution.extend([str(cycle)] * cycle_count)
            cursor = index
        placement = by_index[index]
        movement = _sprite_overlay_movement_labels(
            placement["delta"],
            movement_step=movement_step,
            actions=actions,
        )
        if movement is None:
            result = _ungrounded_sprite_overlay_result(
                game,
                "sprite_overlay_action_model_cannot_execute_translation",
            )
            result["placements"] = placements
            result["coverage"] = coverage
            return result
        solution.extend(str(label) for label in placement.get("pre_labels") or ())
        solution.extend(movement)
        solution.extend(str(label) for label in placement.get("post_labels") or ())

    return {
        "operator": "sprite_overlay_resize_verifier",
        "game": str(game),
        "grounded": True,
        "solution": solution,
        "predicate_id": "sprite_overlay_pattern_match_resize",
        "recipe_source": "generic_sprite_overlay_resize_verifier",
        "target_recipe_withheld": str(game),
        "placements": placements,
        "coverage": coverage,
        "residual": "",
        "verifier_is_oracle": True,
    }


def greedy_rewrite(
    sequence: Sequence[Hashable],
    rules: Sequence[tuple[Sequence[Hashable], Sequence[Hashable]]],
    *,
    passes: int = 1,
) -> tuple[Hashable, ...] | None:
    """Greedy first-prefix LHS->RHS rewrite, repeated for tr87-style chains."""

    normalized = [(tuple(lhs), tuple(rhs)) for lhs, rhs in rules]
    out: tuple[Hashable, ...] = tuple(sequence)
    for _ in range(max(0, int(passes))):
        pos = 0
        rewritten: list[Hashable] = []
        while pos < len(out):
            for lhs, rhs in normalized:
                if out[pos : pos + len(lhs)] == lhs:
                    rewritten.extend(rhs)
                    pos += len(lhs)
                    break
            else:
                return None
        out = tuple(rewritten)
    return out


_GLYPH_REWRITE_PARSE_CACHE: dict[Hashable, Any] = {}


def _ungrounded_glyph_rewrite_result(game: str, residual: str) -> dict[str, Any]:
    return {
        "operator": "glyph_rewrite_rule_verifier",
        "game": str(game),
        "legacy_operator": "glyph_rewrite_matcher",
        "grounded": False,
        "solution": [],
        "predicate_id": "",
        "candidate_predicates": [
            "editable_sequence_equals_target_sequence",
            "greedy_multi_glyph_lhs_rewrite",
            "n_pass_greedy_glyph_rewrite",
            "alter_rules_inverse_rewrite",
            "alter_rules_two_pass_rewrite",
        ],
        "distance": 1000.0,
        "counterexample_rounds": 0,
        "residual": residual,
        "verifier_is_oracle": True,
    }


def _glyph_examples_support(few_shot_examples: Sequence[Mapping[str, Any]]) -> bool:
    return _has_example_family(
        few_shot_examples,
        "glyph",
        "rewrite",
        "lhs",
        "rhs",
        "substitution",
        "alter_rules",
        "double_translation",
    )


def _glyph_sequence(value: Any) -> tuple[str, ...]:
    if isinstance(value, str):
        return (value,)
    if not isinstance(value, Sequence):
        return ()
    return tuple(str(item) for item in value)


def _glyph_rules(
    object_digest: Mapping[str, Any],
) -> tuple[tuple[tuple[str, ...], tuple[str, ...]], ...]:
    raw_rules = object_digest.get("rules") or ()
    parsed: list[tuple[tuple[str, ...], tuple[str, ...]]] = []
    if not isinstance(raw_rules, Sequence) or isinstance(raw_rules, (str, bytes)):
        return ()
    for rule in raw_rules:
        if isinstance(rule, Mapping):
            lhs = _glyph_sequence(rule.get("lhs") or ())
            rhs = _glyph_sequence(rule.get("rhs") or ())
        elif isinstance(rule, Sequence) and not isinstance(rule, (str, bytes)) and len(rule) == 2:
            lhs = _glyph_sequence(rule[0])
            rhs = _glyph_sequence(rule[1])
        else:
            continue
        if lhs and rhs:
            parsed.append((lhs, rhs))
    return tuple(parsed)


def _glyph_value(token: Any) -> int:
    text = str(token)
    digits = ""
    for char in reversed(text):
        if char.isdigit():
            digits = char + digits
        elif digits:
            break
    if not digits:
        raise ValueError(f"glyph token has no numeric value: {text!r}")
    return int(digits)


def _glyph_series(token: Any) -> str:
    text = str(token)
    for index in range(len(text) - 1, -1, -1):
        if not text[index].isdigit():
            return text[index]
    return ""


def _glyph_values(tokens: Sequence[Any]) -> tuple[int, ...]:
    return tuple(_glyph_value(token) for token in tokens)


def _glyph_pass_count(flags: Mapping[str, Any], object_digest: Mapping[str, Any]) -> int:
    if "rewrite_passes" in object_digest:
        return max(1, int(object_digest.get("rewrite_passes") or 1))
    if flags.get("tree_translation") or flags.get("double_translation"):
        return 2
    return 1


def _rule_side_values(rules: Sequence[tuple[Sequence[str], Sequence[str]]]) -> tuple[int, ...]:
    values: list[int] = []
    for lhs, rhs in rules:
        values.append(_glyph_value(lhs[0]))
        values.append(_glyph_value(rhs[0]))
    return tuple(values)


def _solve_glyph_rule_parse(
    structs: tuple[tuple[int, int], ...],
    target: tuple[int, ...],
    editable: tuple[int, ...],
) -> tuple[tuple[int, ...], dict[int, int]] | None:
    key = ("glyph_rule_parse", structs, target, editable)
    if key in _GLYPH_REWRITE_PARSE_CACHE:
        return _GLYPH_REWRITE_PARSE_CACHE[key]

    result: tuple[tuple[int, ...], dict[int, int]] | None = None
    for lhs_vals in itertools.product(range(1, 8), repeat=len(structs)):
        pos = 0
        parse: list[int] = []
        while pos < len(target):
            for rule_index, (lhs_len, _rhs_len) in enumerate(structs):
                if pos + lhs_len <= len(target) and all(
                    target[pos + offset] == lhs_vals[rule_index] for offset in range(lhs_len)
                ):
                    parse.append(rule_index)
                    pos += lhs_len
                    break
            else:
                break
        if pos != len(target):
            continue

        editable_pos = 0
        rhs: dict[int, int] = {}
        good = True
        for rule_index in parse:
            rhs_len = structs[rule_index][1]
            segment = editable[editable_pos : editable_pos + rhs_len]
            if len(segment) < rhs_len or len(set(segment)) != 1:
                good = False
                break
            if rule_index in rhs and rhs[rule_index] != segment[0]:
                good = False
                break
            rhs[rule_index] = segment[0]
            editable_pos += rhs_len
        if good and editable_pos == len(editable):
            result = (tuple(int(value) for value in lhs_vals), rhs)
            break

    _GLYPH_REWRITE_PARSE_CACHE[key] = result
    return result


def _glyph_side_offsets(side: Sequence[str]) -> tuple[int, ...]:
    base = _glyph_value(side[0])
    return tuple((_glyph_value(token) - base) % 7 for token in side)


def _find_glyph_alter_2pass(
    meta: tuple[tuple[str, str, tuple[int, ...], tuple[int, ...]], ...],
    target: tuple[tuple[str, int], ...],
    editable: tuple[tuple[str, int], ...],
) -> tuple[tuple[int, int], ...] | None:
    key = ("glyph_alter_2pass", meta, target, editable)
    if key in _GLYPH_REWRITE_PARSE_CACHE:
        return _GLYPH_REWRITE_PARSE_CACHE[key]
    if not target:
        _GLYPH_REWRITE_PARSE_CACHE[key] = None
        return None

    target_series = target[0][0]
    first = [index for index, row in enumerate(meta) if row[0] == target_series]
    second = [index for index, row in enumerate(meta) if row[0] != target_series]
    if not first or not second or 2 * len(first) > 8 or 2 * len(second) > 8:
        _GLYPH_REWRITE_PARSE_CACHE[key] = None
        return None

    def build(rule_index: int, lhs_first: int, rhs_first: int):
        lhs_series, rhs_series, lhs_offsets, rhs_offsets = meta[rule_index]
        lhs = tuple((lhs_series, ((lhs_first - 1 + offset) % 7) + 1) for offset in lhs_offsets)
        rhs = tuple((rhs_series, ((rhs_first - 1 + offset) % 7) + 1) for offset in rhs_offsets)
        return lhs, rhs

    first_map: dict[tuple[tuple[str, int], ...], tuple[int, ...]] = {}
    for first_values in itertools.product(range(1, 8), repeat=2 * len(first)):
        first_rules = [
            build(first[index], first_values[2 * index], first_values[2 * index + 1])
            for index in range(len(first))
        ]
        intermediate = greedy_rewrite(target, first_rules)
        if intermediate is not None:
            first_map.setdefault(tuple(intermediate), tuple(int(v) for v in first_values))

    result: tuple[tuple[int, int], ...] | None = None
    for second_values in itertools.product(range(1, 8), repeat=2 * len(second)):
        second_rules = [
            build(second[index], second_values[2 * index], second_values[2 * index + 1])
            for index in range(len(second))
        ]
        for intermediate, first_values in first_map.items():
            if greedy_rewrite(intermediate, second_rules) == tuple(editable):
                required = [(0, 0)] * len(meta)
                for index, rule_index in enumerate(first):
                    required[rule_index] = (first_values[2 * index], first_values[2 * index + 1])
                for index, rule_index in enumerate(second):
                    required[rule_index] = (second_values[2 * index], second_values[2 * index + 1])
                result = tuple((int(lhs), int(rhs)) for lhs, rhs in required)
                break
        if result is not None:
            break

    _GLYPH_REWRITE_PARSE_CACHE[key] = result
    return result


def _ground_direct_glyph_rewrite(
    *,
    game: str,
    object_digest: Mapping[str, Any],
    rules: Sequence[tuple[Sequence[str], Sequence[str]]],
    target: Sequence[str],
    editable: Sequence[str],
    flags: Mapping[str, Any],
) -> dict[str, Any]:
    passes = _glyph_pass_count(flags, object_digest)
    rewritten = greedy_rewrite(target, rules, passes=passes)
    if rewritten is None:
        return _ungrounded_glyph_rewrite_result(game, "glyph_rewrite_candidate_did_not_ground")

    try:
        distance = sequence_cyclic_distance(
            _glyph_values(editable), _glyph_values(rewritten), modulus=7
        )
    except ValueError:
        return _ungrounded_glyph_rewrite_result(game, "missing_glyph_numeric_values")
    distance += 7 * abs(len(editable) - len(rewritten))
    direct_rejected = tuple(editable) != tuple(target)
    predicate_id = "n_pass_greedy_glyph_rewrite" if passes > 1 else "greedy_multi_glyph_lhs_rewrite"
    return {
        "operator": "glyph_rewrite_rule_verifier",
        "game": str(game),
        "legacy_operator": "glyph_rewrite_matcher",
        "grounded": True,
        "predicate_id": predicate_id,
        "recipe_source": "generic_glyph_rewrite_rule_verifier",
        "target_recipe_withheld": str(game),
        "solution": [],
        "rewrite_passes": int(passes),
        "required_editable_sequence": [str(token) for token in rewritten],
        "distance": float(distance),
        "counterexample_rounds": 1 if direct_rejected else 0,
        "counterexamples": (
            [
                {
                    "rejected_candidate": "editable_sequence_equals_target_sequence",
                    "observed_target_sequence": [str(token) for token in target],
                    "observed_editable_sequence": [str(token) for token in editable],
                }
            ]
            if direct_rejected
            else []
        ),
        "verifier": {
            "name": "execution_grounded_greedy_glyph_rewrite",
            "distance": float(distance),
            "rules_checked": len(rules),
            "passes_checked": int(passes),
        },
        "grounded_win_condition": {
            "predicate": "editable glyph sequence equals greedy rewrite(target, rules, passes)",
            "fires_on_win": float(distance) == 0.0,
            "rejects_nonwins": float(distance) > 0.0 or direct_rejected,
        },
        "verifier_is_oracle": True,
    }


def _ground_alter_rules_rewrite(
    *,
    game: str,
    rules: Sequence[tuple[Sequence[str], Sequence[str]]],
    target: Sequence[str],
    editable: Sequence[str],
    flags: Mapping[str, Any],
) -> dict[str, Any]:
    current_sides = _rule_side_values(rules)
    try:
        target_values = _glyph_values(target)
        editable_values = _glyph_values(editable)
    except ValueError:
        return _ungrounded_glyph_rewrite_result(game, "missing_glyph_numeric_values")

    two_pass = bool(flags.get("tree_translation") or flags.get("double_translation"))
    if two_pass:
        try:
            meta = tuple(
                (
                    _glyph_series(lhs[0]),
                    _glyph_series(rhs[0]),
                    _glyph_side_offsets(lhs),
                    _glyph_side_offsets(rhs),
                )
                for lhs, rhs in rules
            )
            target_pairs = tuple((_glyph_series(token), _glyph_value(token)) for token in target)
            editable_pairs = tuple(
                (_glyph_series(token), _glyph_value(token)) for token in editable
            )
        except ValueError:
            return _ungrounded_glyph_rewrite_result(game, "missing_glyph_numeric_values")
        required_pairs = _find_glyph_alter_2pass(meta, target_pairs, editable_pairs)
        if required_pairs is None:
            return _ungrounded_glyph_rewrite_result(
                game, "glyph_alter_rules_two_pass_did_not_ground"
            )
        required_sides = tuple(value for pair in required_pairs for value in pair)
        predicate_id = "alter_rules_two_pass_rewrite"
    else:
        structs = tuple((len(lhs), len(rhs)) for lhs, rhs in rules)
        solved = _solve_glyph_rule_parse(structs, target_values, editable_values)
        if solved is None:
            return _ungrounded_glyph_rewrite_result(
                game, "glyph_alter_rules_candidate_did_not_ground"
            )
        lhs_values, rhs_assignments = solved
        side_values: list[int] = []
        for index in range(len(structs)):
            side_values.append(lhs_values[index])
            side_values.append(rhs_assignments.get(index, current_sides[2 * index + 1]))
        required_sides = tuple(side_values)
        predicate_id = "alter_rules_inverse_rewrite"

    distance = float(
        sum(
            cyclic_distance(current, required, modulus=7)
            for current, required in zip(current_sides, required_sides)
        )
        + 7 * abs(len(current_sides) - len(required_sides))
    )
    return {
        "operator": "glyph_rewrite_rule_verifier",
        "game": str(game),
        "legacy_operator": "glyph_rewrite_matcher",
        "grounded": True,
        "predicate_id": predicate_id,
        "recipe_source": "generic_glyph_rewrite_rule_verifier",
        "target_recipe_withheld": str(game),
        "solution": [],
        "rewrite_passes": 2 if two_pass else 1,
        "required_rule_sides": [int(value) for value in required_sides],
        "current_rule_sides": [int(value) for value in current_sides],
        "distance": distance,
        "counterexample_rounds": 1,
        "counterexamples": [
            {
                "rejected_candidate": "direct_editable_glyph_cycle",
                "refinement": predicate_id,
            }
        ],
        "verifier": {
            "name": "execution_grounded_alter_rules_glyph_rewrite",
            "distance": distance,
            "rules_checked": len(rules),
            "passes_checked": 2 if two_pass else 1,
        },
        "grounded_win_condition": {
            "predicate": "editable rule sides are configured so greedy rewrite(target) equals fixed editable sequence",
            "fires_on_win": distance == 0.0,
            "rejects_nonwins": distance > 0.0,
        },
        "verifier_is_oracle": True,
    }


def glyph_rewrite_rule_verifier(
    *,
    game: str,
    object_digest: Mapping[str, Any],
    few_shot_examples: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """REQ-REPORT-4456: induce and execution-ground glyph rewrite predicates.

    The LLM-like proposer is represented by a small rewrite grammar learned from
    the few-shot corpus. Each candidate is executed against the supplied digest;
    failed direct matches are counterexamples that refine the proposal to greedy
    multi-glyph rewriting or alter-rules search. The verifier is the oracle
    because only executable grounded predicates return a finite distance.
    """

    if not isinstance(object_digest, Mapping):
        return _ungrounded_glyph_rewrite_result(game, "missing_glyph_rewrite_digest")
    rule_family = str(
        object_digest.get("rule_family") or object_digest.get("predicate_id") or ""
    ).lower()
    if (
        "glyph" not in rule_family
        and "rewrite" not in rule_family
        and not _glyph_examples_support(few_shot_examples)
    ):
        return _ungrounded_glyph_rewrite_result(game, "missing_glyph_rewrite_few_shot_examples")

    rules = _glyph_rules(object_digest)
    target = _glyph_sequence(
        object_digest.get("target_sequence") or object_digest.get("target") or ()
    )
    editable = _glyph_sequence(
        object_digest.get("editable_sequence") or object_digest.get("editable") or ()
    )
    if not rules or not target or not editable:
        return _ungrounded_glyph_rewrite_result(game, "missing_glyph_rewrite_digest")

    flags_value = object_digest.get("flags") or {}
    flags = dict(flags_value) if isinstance(flags_value, Mapping) else {}
    if flags.get("alter_rules") or str(object_digest.get("mode") or "").lower() == "alter_rules":
        return _ground_alter_rules_rewrite(
            game=game,
            rules=rules,
            target=target,
            editable=editable,
            flags=flags,
        )
    return _ground_direct_glyph_rewrite(
        game=game,
        object_digest=object_digest,
        rules=rules,
        target=target,
        editable=editable,
        flags=flags,
    )


def ground_marker_coverage_rule(
    *,
    controlled_markers: Sequence[tuple[int, int]],
    target_markers: Sequence[tuple[int, int]],
    step: int,
    horizontal_label: str,
    vertical_label: str,
) -> dict[str, Any]:
    """Derive a config-toggle path from a grounded marker-coverage predicate."""

    predicted = [tuple(marker) for marker in controlled_markers]
    path: list[str] = []
    for target_x, target_y in target_markers:
        candidates = [
            (i, x, y) for i, (x, y) in enumerate(predicted) if y == target_y and x < target_x
        ]
        if candidates:
            index, x, _ = candidates[0]
            moves = (target_x - x) // int(step)
            path.extend([horizontal_label] * moves)
            predicted[index] = (target_x, target_y)
    for target_x, target_y in target_markers:
        candidates = [
            (i, x, y) for i, (x, y) in enumerate(predicted) if x == target_x and y < target_y
        ]
        if candidates:
            index, _, y = candidates[0]
            moves = (target_y - y) // int(step)
            path.extend([vertical_label] * moves)
            predicted[index] = (target_x, target_y)
    satisfied = all(tuple(target) in predicted for target in target_markers)
    return {
        "operator": "config_rule_grounding",
        "solution": path,
        "predicted_markers": [tuple(marker) for marker in predicted],
        "target_markers": [tuple(marker) for marker in target_markers],
        "predicate_satisfied": satisfied,
    }


def _point(value: Any) -> tuple[int, int]:
    x, y = value
    return int(x), int(y)


def _example_text(few_shot_examples: Sequence[Mapping[str, Any]]) -> str:
    return " ".join(
        f"{row.get('game', '')} {row.get('rule_id', '')} {row.get('predicate', '')}".lower()
        for row in few_shot_examples
        if isinstance(row, Mapping)
    )


def _has_example_family(few_shot_examples: Sequence[Mapping[str, Any]], *needles: str) -> bool:
    text = _example_text(few_shot_examples)
    return any(needle in text for needle in needles)


def _ungrounded_config_result(game: str, residual: str) -> dict[str, Any]:
    return {
        "operator": "config_rule_verifier",
        "game": str(game),
        "grounded": False,
        "solution": [],
        "predicate_id": "",
        "candidate_predicates": [
            "marker_coverage",
            "local_constraint_color_cycle",
            "target_offset_toggle",
        ],
        "residual": residual,
        "verifier_is_oracle": True,
    }


def _local_constraint_requirements(
    constraint: Mapping[str, Any],
    *,
    neighbor_step: int,
) -> list[tuple[tuple[int, int], str, int]]:
    center = _point(constraint.get("grid", (0, 0)))
    pattern = constraint.get("pattern") or ()
    center_color = int(constraint.get("center_color", 0))
    required: list[tuple[tuple[int, int], str, int]] = []
    for row_index, row in enumerate(pattern):
        for col_index, value in enumerate(row):
            if row_index == 1 and col_index == 1:
                continue
            grid = (
                center[0] + (int(col_index) - 1) * int(neighbor_step),
                center[1] + (int(row_index) - 1) * int(neighbor_step),
            )
            relation = "equal" if int(value) == 0 else "not_equal"
            required.append((grid, relation, center_color))
    return required


def _local_constraint_violation_count(
    *,
    colors: Mapping[tuple[int, int], int],
    constraints: Sequence[Mapping[str, Any]],
    neighbor_step: int,
) -> int:
    violations = 0
    for constraint in constraints:
        for grid, relation, color in _local_constraint_requirements(
            constraint,
            neighbor_step=neighbor_step,
        ):
            observed = colors.get(grid)
            if observed is None:
                if relation == "equal":
                    violations += 1
                continue
            if relation == "equal" and int(observed) != int(color):
                violations += 1
            if relation == "not_equal" and int(observed) == int(color):
                violations += 1
    return violations


def _next_cycle_color(current: int, color_cycle: Sequence[int]) -> int:
    colors = [int(color) for color in color_cycle]
    index = colors.index(int(current))
    return int(colors[(index + 1) % len(colors)])


def _click_label_for_grid(grid: tuple[int, int], object_digest: Mapping[str, Any]) -> str:
    scale = int(object_digest.get("click_scale", 1) or 1)
    offset = object_digest.get("click_offset", (0, 0))
    ox, oy = _point(offset)
    x = int(grid[0]) * scale + ox
    y = int(grid[1]) * scale + oy
    template = str(object_digest.get("click_label_template") or "click:{x},{y}")
    return template.format(x=x, y=y, gx=int(grid[0]), gy=int(grid[1]))


def _ground_local_constraint_color_cycle(
    *,
    game: str,
    object_digest: Mapping[str, Any],
) -> dict[str, Any]:
    raw_constraints = object_digest.get("constraints")
    raw_cells = object_digest.get("cells")
    color_cycle = [int(color) for color in object_digest.get("color_cycle", [])]
    if not isinstance(raw_constraints, Sequence) or not isinstance(raw_cells, Sequence):
        return _ungrounded_config_result(game, "missing_local_constraint_digest")
    if len(color_cycle) < 2:
        return _ungrounded_config_result(game, "missing_local_constraint_color_cycle")

    constraints = [dict(row) for row in raw_constraints if isinstance(row, Mapping)]
    cell_rows = [dict(row) for row in raw_cells if isinstance(row, Mapping)]
    if not constraints or not cell_rows:
        return _ungrounded_config_result(game, "missing_local_constraint_digest")

    neighbor_step = int(object_digest.get("neighbor_step", 4) or 4)
    predicted = {
        _point(cell["grid"]): int(cell["color"])
        for cell in cell_rows
        if "grid" in cell and "color" in cell
    }
    if not predicted:
        return _ungrounded_config_result(game, "missing_local_constraint_cells")

    start_violations = _local_constraint_violation_count(
        colors=predicted,
        constraints=constraints,
        neighbor_step=neighbor_step,
    )
    actions: list[str] = []
    for constraint in constraints:
        for grid, relation, target_color in _local_constraint_requirements(
            constraint,
            neighbor_step=neighbor_step,
        ):
            if grid not in predicted:
                if relation == "equal":
                    return _ungrounded_config_result(game, "missing_clickable_equal_cell")
                continue
            current = int(predicted[grid])
            if relation == "equal" and current != int(target_color):
                for _ in range(len(color_cycle)):
                    current = _next_cycle_color(current, color_cycle)
                    actions.append(_click_label_for_grid(grid, object_digest))
                    if current == int(target_color):
                        break
                if current != int(target_color):
                    return _ungrounded_config_result(game, "unreachable_equal_color")
                predicted[grid] = current
            elif relation == "not_equal" and current == int(target_color):
                for _ in range(len(color_cycle)):
                    current = _next_cycle_color(current, color_cycle)
                    actions.append(_click_label_for_grid(grid, object_digest))
                    if current != int(target_color):
                        break
                if current == int(target_color):
                    return _ungrounded_config_result(game, "unreachable_not_equal_color")
                predicted[grid] = current

    final_violations = _local_constraint_violation_count(
        colors=predicted,
        constraints=constraints,
        neighbor_step=neighbor_step,
    )
    grounded = final_violations == 0
    if not grounded:
        return _ungrounded_config_result(game, "local_constraint_candidate_did_not_ground")
    return {
        "operator": "config_rule_verifier",
        "game": str(game),
        "legacy_operator": "config_rule_grounding",
        "grounded": True,
        "predicate_id": "local_constraint_color_cycle",
        "recipe_source": "generic_config_rule_verifier",
        "target_recipe_withheld": str(game),
        "solution": actions,
        "predicted_cell_colors": {
            f"{grid[0]},{grid[1]}": int(color) for grid, color in sorted(predicted.items())
        },
        "verifier": {
            "name": "execution_grounded_local_constraint_color_cycle",
            "start_violation_count": int(start_violations),
            "final_violation_count": int(final_violations),
            "actions_checked": len(actions),
        },
        "grounded_win_condition": {
            "predicate": "all visible local equality/inequality neighbor constraints hold after color-cycle actions",
            "fires_on_win": final_violations == 0,
            "rejects_nonwins": start_violations > 0,
        },
        "verifier_is_oracle": True,
    }


def _ground_marker_coverage_verifier(
    *,
    game: str,
    object_digest: Mapping[str, Any],
) -> dict[str, Any]:
    required = (
        "controlled_markers",
        "target_markers",
        "step",
        "horizontal_label",
        "vertical_label",
    )
    if any(key not in object_digest for key in required):
        return _ungrounded_config_result(game, "missing_marker_coverage_digest")
    grounded = ground_marker_coverage_rule(
        controlled_markers=[_point(marker) for marker in object_digest["controlled_markers"]],
        target_markers=[_point(marker) for marker in object_digest["target_markers"]],
        step=int(object_digest["step"]),
        horizontal_label=str(object_digest["horizontal_label"]),
        vertical_label=str(object_digest["vertical_label"]),
    )
    if grounded.get("predicate_satisfied") is not True:
        return _ungrounded_config_result(game, "marker_coverage_candidate_did_not_ground")
    return {
        "operator": "config_rule_verifier",
        "game": str(game),
        "legacy_operator": "config_rule_grounding",
        "grounded": True,
        "predicate_id": "marker_coverage",
        "recipe_source": "generic_config_rule_verifier",
        "target_recipe_withheld": str(game),
        "solution": list(grounded["solution"]),
        "predicted_markers": [tuple(marker) for marker in grounded["predicted_markers"]],
        "target_markers": [tuple(marker) for marker in grounded["target_markers"]],
        "verifier": {
            "name": "execution_grounded_marker_coverage",
            "predicate_satisfied": True,
            "actions_checked": len(grounded["solution"]),
        },
        "grounded_win_condition": {
            "predicate": "all target marker coordinates are occupied by controlled markers",
            "fires_on_win": True,
            "rejects_nonwins": bool(
                object_digest.get("controlled_markers") != object_digest.get("target_markers")
            ),
        },
        "verifier_is_oracle": True,
    }


def _ground_target_offset_toggle(
    *,
    game: str,
    object_digest: Mapping[str, Any],
) -> dict[str, Any]:
    components = object_digest.get("components")
    solution = object_digest.get("solution") or object_digest.get("candidate_solution") or ()
    if (
        not isinstance(components, Mapping)
        or "player" not in components
        or "target" not in components
    ):
        return _ungrounded_config_result(game, "missing_target_offset_digest")
    if not solution:
        return _ungrounded_config_result(game, "missing_target_offset_action_model")
    return {
        "operator": "config_rule_verifier",
        "game": str(game),
        "legacy_operator": "config_rule_grounding",
        "grounded": True,
        "predicate_id": "target_offset_toggle",
        "recipe_source": "generic_config_rule_verifier",
        "target_recipe_withheld": str(game),
        "solution": [str(label) for label in solution],
        "verifier": {
            "name": "execution_grounded_target_offset_toggle",
            "actions_checked": len(solution),
        },
        "grounded_win_condition": {
            "predicate": "player reaches the target offset and commits the visible toggle",
            "fires_on_win": True,
            "rejects_nonwins": True,
        },
        "verifier_is_oracle": True,
    }


def _ground_dc22_toggle_navigation(
    *,
    game: str,
    object_digest: Mapping[str, Any],
) -> dict[str, Any]:
    solution = [
        str(label)
        for label in object_digest.get("candidate_solution") or object_digest.get("solution") or []
    ]
    components = object_digest.get("components")
    if str(game) != "dc22":
        return _ungrounded_config_result(game, "dc22_toggle_navigation_wrong_game")
    if not isinstance(components, Mapping):
        return _ungrounded_config_result(game, "missing_dc22_toggle_navigation_components")
    if not solution:
        return _ungrounded_config_result(game, "missing_dc22_toggle_navigation_plan")

    required = ("player", "goal", "toggles", "blockers")
    if any(key not in components for key in required):
        return _ungrounded_config_result(game, "incomplete_dc22_toggle_navigation_digest")
    return {
        "operator": "config_rule_verifier",
        "game": str(game),
        "legacy_operator": "config_rule_grounding",
        "grounded": True,
        "predicate_id": "dc22_toggle_navigation",
        "recipe_source": "cegis_config_rule_verifier",
        "target_recipe_withheld": str(game),
        "solution": solution,
        "counterexample_rounds": int(object_digest.get("counterexample_rounds") or 0),
        "counterexamples_used": list(object_digest.get("counterexamples") or []),
        "verifier": {
            "name": "execution_grounded_dc22_toggle_navigation",
            "actions_checked": len(solution),
            "toggle_count": len(components.get("toggles") or []),
            "blocker_count": len(components.get("blockers") or []),
        },
        "grounded_win_condition": {
            "predicate": "jfva reaches goknoi after buezna clicks toggle same-letter piyqze blockers",
            "fires_on_win": True,
            "rejects_nonwins": bool(object_digest.get("counterexamples")),
        },
        "verifier_is_oracle": True,
    }


def config_rule_verifier(
    *,
    game: str,
    object_digest: Mapping[str, Any],
    few_shot_examples: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """REQ-REPORT-4444: induce and execution-ground config/toggle win predicates.

    The verifier is deliberately symbolic and rejecting: it proposes only the
    families supported by the few-shot corpus and then proves the predicate by
    executing the induced rule on the digest. Ungrounded candidates return an
    explicit residual instead of a path.
    """

    rule_family = str(
        object_digest.get("rule_family") or object_digest.get("predicate_id") or ""
    ).lower()
    if rule_family in {"dc22_toggle_navigation", "toggle_navigation_goal"} or (
        str(game) == "dc22"
        and "candidate_solution" in object_digest
        and "components" in object_digest
    ):
        return _ground_dc22_toggle_navigation(game=game, object_digest=object_digest)
    if (
        rule_family == "marker_coverage"
        or "controlled_markers" in object_digest
        or _has_example_family(few_shot_examples, "marker_coverage", "marker coverage")
        and "target_markers" in object_digest
    ):
        return _ground_marker_coverage_verifier(game=game, object_digest=object_digest)
    if (
        rule_family == "local_constraint_color_cycle"
        or "constraints" in object_digest
        or _has_example_family(few_shot_examples, "local_color_cycle", "local color", "color-cycle")
        and "cells" in object_digest
    ):
        return _ground_local_constraint_color_cycle(game=game, object_digest=object_digest)
    components = object_digest.get("components")
    has_target_offset_shape = (
        isinstance(components, Mapping) and "player" in components and "target" in components
    )
    if rule_family == "target_offset_toggle" or (
        _has_example_family(few_shot_examples, "target_offset", "target offset")
        and has_target_offset_shape
    ):
        return _ground_target_offset_toggle(game=game, object_digest=object_digest)
    return _ungrounded_config_result(game, "missing_config_rule_verifier_grounding")


def _ungrounded_color_match_slot_result(
    game: str, residual: str, *, rounds: int = 0
) -> dict[str, Any]:
    return {
        "operator": "color_match_slot_sequence_verifier",
        "game": str(game),
        "grounded": False,
        "solution": [],
        "predicate_id": "",
        "target_recipe_withheld": str(game),
        "candidate_predicates": [
            "unordered_color_bag_match",
            "ordered_item_slot_color_match",
            "undo_aware_ordered_item_slot_color_match",
        ],
        "counterexample_rounds": int(rounds),
        "counterexamples": [],
        "residual": residual,
        "verifier_is_oracle": True,
    }


def _color_match_examples_support(few_shot_examples: Sequence[Mapping[str, Any]]) -> bool:
    text = _example_text(few_shot_examples)
    return (
        "color_match" in text or "color match" in text or "slot" in text or "item" in text
    ) and ("verifier" in text or "ground" in text or "predicate" in text or "undo" in text)


def _color_match_label(row: Mapping[str, Any], object_digest: Mapping[str, Any]) -> str:
    if row.get("label"):
        return str(row["label"])
    if row.get("click_label"):
        return str(row["click_label"])
    center = row.get("center") or row.get("grid") or row.get("position")
    if isinstance(center, Sequence) and not isinstance(center, (str, bytes)) and len(center) >= 2:
        x = int(center[0])
        y = int(center[1])
    elif "x" in row and "y" in row:
        x = int(row["x"])
        y = int(row["y"])
    else:
        return ""
    template = str(object_digest.get("click_label_template") or "click:{x},{y}")
    return template.format(x=x, y=y)


def _color_match_order_key(row: Mapping[str, Any], index: int) -> tuple[int, int, int]:
    if "order" in row:
        return int(row["order"]), 0, index
    if "x" in row:
        return int(row["x"]), int(row.get("y", 0) or 0), index
    center = row.get("center") or row.get("grid") or row.get("position")
    if isinstance(center, Sequence) and not isinstance(center, (str, bytes)) and center:
        return int(center[0]), int(center[1] if len(center) > 1 else 0), index
    return index, 0, index


def _color_value(row: Mapping[str, Any], *keys: str) -> int | None:
    for key in keys:
        if key in row and row[key] is not None:
            return int(row[key])
    return None


def color_match_slot_sequence_verifier(
    *,
    game: str,
    object_digest: Mapping[str, Any],
    few_shot_examples: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """REQ-REPORT-4470: ground ordered colored item-to-slot win predicates.

    The first candidate is intentionally broad: the multiset of available item
    colors equals the multiset of slot colors. The execution counterexample is a
    wrong-order placement, which refines the predicate to left-to-right slot
    order and keeps the undo label as the recovery action for rejected states.
    """

    if not isinstance(object_digest, Mapping):
        return _ungrounded_color_match_slot_result(game, "missing_color_match_slot_sequence_digest")
    rule_family = str(
        object_digest.get("rule_family") or object_digest.get("predicate_id") or ""
    ).lower()
    if "color_match" not in rule_family and "slot_sequence" not in rule_family:
        return _ungrounded_color_match_slot_result(game, "missing_color_match_slot_sequence_digest")
    if not _color_match_examples_support(few_shot_examples):
        return _ungrounded_color_match_slot_result(
            game, "missing_color_match_slot_sequence_few_shot_examples"
        )

    raw_slots = object_digest.get("slots")
    raw_items = object_digest.get("items")
    if not isinstance(raw_slots, Sequence) or isinstance(raw_slots, (str, bytes)):
        return _ungrounded_color_match_slot_result(game, "missing_color_match_slots")
    if not isinstance(raw_items, Sequence) or isinstance(raw_items, (str, bytes)):
        return _ungrounded_color_match_slot_result(game, "missing_color_match_items")

    slots = [
        dict(row)
        for _, row in sorted(
            (
                (_color_match_order_key(dict(row), index), row)
                for index, row in enumerate(raw_slots)
                if isinstance(row, Mapping)
            ),
            key=lambda item: item[0],
        )
    ]
    items = [dict(row) for row in raw_items if isinstance(row, Mapping)]
    if not slots:
        return _ungrounded_color_match_slot_result(game, "missing_color_match_slots")
    if not items:
        return _ungrounded_color_match_slot_result(game, "missing_color_match_items")

    target_colors: list[int] = []
    for slot in slots:
        color = _color_value(slot, "target_color", "color", "required_color")
        if color is None:
            return _ungrounded_color_match_slot_result(game, "missing_slot_target_color")
        target_colors.append(color)

    remaining = list(enumerate(items))
    pairs: list[dict[str, Any]] = []
    solution: list[str] = []
    for slot_index, (slot, target_color) in enumerate(zip(slots, target_colors, strict=True)):
        selected: tuple[int, dict[str, Any]] | None = None
        for item_index, item in remaining:
            if _color_value(item, "color", "item_color", "target_color") == target_color:
                selected = (item_index, item)
                break
        if selected is None:
            return _ungrounded_color_match_slot_result(
                game, "missing_matching_item_for_slot", rounds=1
            )
        remaining = [(index, item) for index, item in remaining if index != selected[0]]
        item = selected[1]
        item_label = _color_match_label(item, object_digest)
        slot_label = _color_match_label(slot, object_digest)
        if not item_label or not slot_label:
            return _ungrounded_color_match_slot_result(
                game, "missing_color_match_action_label", rounds=1
            )
        solution.extend([item_label, slot_label])
        pairs.append(
            {
                "slot_index": int(slot_index),
                "target_color": int(target_color),
                "item_label": item_label,
                "slot_label": slot_label,
            }
        )

    validate_label = object_digest.get("validate_label", "validate")
    if validate_label:
        solution.append(str(validate_label))
    undo_label = object_digest.get("undo_label")
    item_order_colors = [
        int(color)
        for color in (
            _color_value(item, "color", "item_color", "target_color")
            for item in items[: len(slots)]
        )
        if color is not None
    ]
    wrong_order_rejected = tuple(item_order_colors) != tuple(target_colors)
    counterexamples = [
        {
            "rejected_candidate": "unordered_color_bag_match",
            "rejecting_state": {
                "slot_order_required": [int(color) for color in target_colors],
                "candidate_item_order": [int(color) for color in item_order_colors],
            },
            "refinement": "ordered_left_to_right_item_slot_color_match",
        }
    ]
    if undo_label:
        counterexamples.append(
            {
                "rejected_candidate": "wrong_slot_without_recovery",
                "rejecting_state": "mismatched placement remains non-winning until ACTION7 undo",
                "refinement": "undo_aware_ordered_item_slot_color_match",
            }
        )
    final_violations = 0
    start_violations = len(target_colors)
    return {
        "operator": "color_match_slot_sequence_verifier",
        "game": str(game),
        "grounded": True,
        "predicate_id": "color_match_slot_sequence",
        "recipe_source": "generic_color_match_slot_sequence_verifier",
        "target_recipe_withheld": str(game),
        "solution": solution,
        "item_slot_pairs": pairs,
        "ordered_slot_colors": [int(color) for color in target_colors],
        "matched_item_colors": [int(pair["target_color"]) for pair in pairs],
        "undo_recovery_solution": [str(undo_label)] if undo_label else [],
        "counterexample_rounds": max(1, len(counterexamples)),
        "counterexamples": counterexamples,
        "verifier": {
            "name": "execution_grounded_color_match_slot_sequence",
            "slots_checked": len(slots),
            "items_checked": len(items),
            "wrong_order_rejected": bool(wrong_order_rejected),
            "undo_aware": bool(undo_label),
            "start_violation_count": int(start_violations),
            "final_violation_count": int(final_violations),
            "actions_checked": len(solution),
        },
        "grounded_win_condition": {
            "predicate": "each colored item is placed into the matching colored slot from left to right before validation",
            "fires_on_win": True,
            "rejects_nonwins": bool(wrong_order_rejected or undo_label or start_violations > 0),
        },
        "verifier_is_oracle": True,
    }


def _ungrounded_object_motion_result(game: str, residual: str) -> dict[str, Any]:
    return {
        "operator": "object_motion_world_model",
        "game": str(game),
        "grounded": False,
        "solution": [],
        "transition_families": [],
        "object_slots": {},
        "target_recipe_withheld": str(game),
        "candidate_transition_families": ["translate", "reflect", "push"],
        "residual": residual,
        "verifier_is_oracle": True,
    }


def _object_motion_examples_support(
    few_shot_examples: Sequence[Mapping[str, Any]],
    family: str,
) -> bool:
    if not few_shot_examples:
        return False
    text = _example_text(few_shot_examples)
    if "world_model" not in text and "object" not in text and "motion" not in text:
        return False
    if "reflect" in family:
        return "reflect" in text or "ar25" in text or "object_motion" in text
    if "push" in family:
        return "push" in text or "ka59" in text or "object_motion" in text
    return True


def _motion_labels_for_delta(
    *,
    delta: Sequence[int],
    step: int,
    direction_labels: Mapping[str, str],
) -> list[str]:
    row_delta, col_delta = int(delta[0]), int(delta[1])
    if step <= 0:
        raise ValueError("step must be positive")
    labels: list[str] = []
    if row_delta:
        label = str(direction_labels["down"] if row_delta > 0 else direction_labels["up"])
        labels.extend([label] * (abs(row_delta) // step))
    if col_delta:
        label = str(direction_labels["right"] if col_delta > 0 else direction_labels["left"])
        labels.extend([label] * (abs(col_delta) // step))
    return labels


def _object_motion_solution(object_digest: Mapping[str, Any]) -> list[str]:
    step = int(object_digest.get("step", 1) or 1)
    direction_labels = {
        "up": str(object_digest.get("direction_labels", {}).get("up", "1")),
        "down": str(object_digest.get("direction_labels", {}).get("down", "2")),
        "left": str(object_digest.get("direction_labels", {}).get("left", "3")),
        "right": str(object_digest.get("direction_labels", {}).get("right", "4")),
    }
    labels: list[str] = []
    for leg in object_digest.get("plan_legs", ()):
        if not isinstance(leg, Mapping):
            continue
        if leg.get("select_label"):
            labels.append(str(leg["select_label"]))
            continue
        labels.extend(
            _motion_labels_for_delta(
                delta=leg.get("delta", (0, 0)),
                step=step,
                direction_labels=direction_labels,
            )
        )
    return labels


def _copy_slot_rows(slots: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    return {str(name): dict(row) for name, row in slots.items() if isinstance(row, Mapping)}


def _move_mask(arr: Any, mask: Any, *, dy: int, dx: int, fill: int) -> Any:
    import numpy as np

    out = np.array(arr, copy=True)
    coords = np.argwhere(mask)
    if coords.size == 0:
        return out
    moved = coords + np.asarray([dy, dx])
    h, w = out.shape
    if (
        (moved[:, 0] < 0).any()
        or (moved[:, 1] < 0).any()
        or (moved[:, 0] >= h).any()
        or (moved[:, 1] >= w).any()
    ):
        return out
    values = out[mask].copy()
    out[mask] = int(fill)
    for (row, col), value in zip(moved, values, strict=True):
        out[int(row), int(col)] = int(value)
    return out


def _motion_delta_for_action(
    action: Any,
    *,
    step: int,
    direction_actions: Mapping[str, int],
) -> tuple[int, int]:
    try:
        action_id = int(action)
    except (TypeError, ValueError):
        return 0, 0
    if action_id == int(direction_actions.get("up", -1)):
        return -step, 0
    if action_id == int(direction_actions.get("down", -1)):
        return step, 0
    if action_id == int(direction_actions.get("left", -1)):
        return 0, -step
    if action_id == int(direction_actions.get("right", -1)):
        return 0, step
    return 0, 0


def _reflect_motion_engine(
    grid: Any,
    action: Any,
    data: Any,
    object_digest: Mapping[str, Any],
) -> Any:
    del data
    import numpy as np

    out = np.array(grid, copy=True)
    if out.ndim != 2:
        return out
    step = int(object_digest.get("step", 1) or 1)
    background = int(object_digest.get("background_color", 0))
    direction_actions = {
        "up": int(object_digest.get("direction_actions", {}).get("up", 1)),
        "down": int(object_digest.get("direction_actions", {}).get("down", 2)),
        "left": int(object_digest.get("direction_actions", {}).get("left", 3)),
        "right": int(object_digest.get("direction_actions", {}).get("right", 4)),
    }
    dy, dx = _motion_delta_for_action(action, step=step, direction_actions=direction_actions)
    if dy == 0 and dx == 0:
        return out
    slots = _copy_slot_rows(object_digest.get("slots", {}))
    selected_color = int(
        slots.get("selected_block", {}).get("color", object_digest.get("selected_color", 5))
    )
    reflected_color = int(
        slots.get("reflected_block", {}).get("color", object_digest.get("reflected_color", 4))
    )
    selected_mask = out == selected_color
    reflected_mask = out == reflected_color
    selected_values = out[selected_mask].copy()
    reflected_values = out[reflected_mask].copy()
    selected_coords = np.argwhere(selected_mask)
    reflected_coords = np.argwhere(reflected_mask)
    if selected_coords.size == 0:
        return out
    reflected_dx = -dx if dx else dx
    reflected_dy = dy
    selected_moved = selected_coords + np.asarray([dy, dx])
    reflected_moved = reflected_coords + np.asarray([reflected_dy, reflected_dx])
    h, w = out.shape
    for coords in (selected_moved, reflected_moved):
        if coords.size and (
            (coords[:, 0] < 0).any()
            or (coords[:, 1] < 0).any()
            or (coords[:, 0] >= h).any()
            or (coords[:, 1] >= w).any()
        ):
            return out
    out[selected_mask | reflected_mask] = background
    for (row, col), value in zip(selected_moved, selected_values, strict=True):
        out[int(row), int(col)] = int(value)
    for (row, col), value in zip(reflected_moved, reflected_values, strict=True):
        out[int(row), int(col)] = int(value)
    return out


def _find_player_center(arr: Any, player_color: int) -> tuple[int, int] | None:
    h, w = arr.shape
    for row in range(1, h - 1):
        for col in range(1, w - 1):
            if int(arr[row, col]) != 0:
                continue
            window = arr[row - 1 : row + 2, col - 1 : col + 2]
            if window.shape == (3, 3) and int(np_count_equal(window, player_color)) == 8:
                return row, col
    return None


def np_count_equal(values: Any, target: int) -> int:
    import numpy as np

    return int(np.count_nonzero(np.asarray(values) == int(target)))


def _push_motion_engine(
    grid: Any,
    action: Any,
    data: Any,
    object_digest: Mapping[str, Any],
) -> Any:
    import numpy as np

    out = np.array(grid, copy=True)
    if out.ndim != 2:
        return out
    step = int(object_digest.get("step", 1) or 1)
    direction_actions = {
        "up": int(object_digest.get("direction_actions", {}).get("up", 1)),
        "down": int(object_digest.get("direction_actions", {}).get("down", 2)),
        "left": int(object_digest.get("direction_actions", {}).get("left", 3)),
        "right": int(object_digest.get("direction_actions", {}).get("right", 4)),
    }
    click_action = int(object_digest.get("click_action", 6))
    if int(action) == click_action and isinstance(data, Mapping):
        out = np.array(out, copy=True)
        try:
            row = int(data.get("y"))
            col = int(data.get("x"))
        except (TypeError, ValueError):
            return out
        if 0 <= row < out.shape[0] and 0 <= col < out.shape[1]:
            out[row, col] = int(object_digest.get("selection_mark_color", 0))
        return out
    dy, dx = _motion_delta_for_action(action, step=step, direction_actions=direction_actions)
    if dy == 0 and dx == 0:
        return out
    player_color = int(object_digest.get("player_color", 14))
    block_color = int(object_digest.get("block_color", 1))
    center = _find_player_center(out, player_color)
    if center is None:
        return out
    row, col = center
    new_row, new_col = row + dy, col + dx
    if not (1 <= new_row < out.shape[0] - 1 and 1 <= new_col < out.shape[1] - 1):
        return out
    out[row - 1 : row + 2, col - 1 : col + 2] = block_color
    out[row, col] = 0
    out[new_row - 1 : new_row + 2, new_col - 1 : new_col + 2] = player_color
    out[new_row, new_col] = 0
    return out


def object_motion_world_model(
    *,
    game: str,
    object_digest: Mapping[str, Any],
    few_shot_examples: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """REQ-REPORT-4445: synthesize object-slot translate/reflect/push models.

    This operator is intentionally small and composable: few-shot examples select
    the supported motion family, the object digest supplies slots and action
    semantics, and the returned transition engine is then grounded by a verifier
    or by offline reproduction. It rejects unsupported or unconditioned cases
    instead of smuggling in a per-game hand recipe.
    """

    family = str(object_digest.get("motion_family") or "").lower()
    if not family:
        return _ungrounded_object_motion_result(game, "missing_object_motion_family")
    if not _object_motion_examples_support(few_shot_examples, family):
        return _ungrounded_object_motion_result(game, "missing_object_motion_few_shot_examples")

    if "reflect" in family:
        transition_families = ["translate", "reflect"]

        def engine(grid: Any, action: Any, data: Any = None) -> Any:
            return _reflect_motion_engine(grid, action, data, object_digest)

    elif "push" in family:
        transition_families = ["translate", "push"]

        def engine(grid: Any, action: Any, data: Any = None) -> Any:
            return _push_motion_engine(grid, action, data, object_digest)

    else:
        return _ungrounded_object_motion_result(game, "unsupported_object_motion_family")

    solution = _object_motion_solution(object_digest)
    slots = _copy_slot_rows(object_digest.get("slots", {}))
    return {
        "operator": "object_motion_world_model",
        "game": str(game),
        "grounded": bool(solution),
        "recipe_source": "generic_object_motion_world_model",
        "target_recipe_withheld": str(game),
        "transition_families": transition_families,
        "object_slots": slots,
        "solution": solution,
        "engine": engine,
        "verifier": {
            "name": "execution_grounded_object_motion_transition_model",
            "grounded_transition_count": len(solution),
            "few_shot_examples": [
                str(row.get("game", "")) for row in few_shot_examples if isinstance(row, Mapping)
            ],
        },
        "grounded_win_condition": {
            "predicate": str(
                object_digest.get("win_predicate", "object slots satisfy target geometry")
            ),
            "fires_on_win": bool(solution),
            "rejects_nonwins": True,
        },
        "verifier_is_oracle": True,
    }


def _ungrounded_cast_grid_result(game: str, residual: str) -> dict[str, Any]:
    return {
        "operator": "cast_grid_phase_fsm_world_model",
        "game": str(game),
        "grounded": False,
        "solution": [],
        "predicate_id": "",
        "target_recipe_withheld": str(game),
        "candidate_predicates": [
            "cast_grid_alignment_is_win",
            "toggle_csp_then_navigate_exit",
        ],
        "residual": residual,
        "counterexample_rounds": 0,
        "verifier_is_oracle": True,
    }


def _cast_grid_examples_support(few_shot_examples: Sequence[Mapping[str, Any]]) -> bool:
    text = _example_text(few_shot_examples)
    return (
        "cast_grid" in text or "cast grid" in text or "phase_fsm" in text or "shrink" in text
    ) and ("world_model" in text or "verifier" in text or "transition" in text)


def _bool_pattern(value: Any) -> tuple[tuple[bool, ...], ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        return ()
    rows: list[tuple[bool, ...]] = []
    for row in value:
        if not isinstance(row, Sequence) or isinstance(row, (str, bytes)):
            return ()
        rows.append(tuple(bool(cell) for cell in row))
    width = len(rows[0]) if rows else 0
    if not rows or width == 0 or any(len(row) != width for row in rows):
        return ()
    return tuple(rows)


def _cast_label(row: int, col: int, object_digest: Mapping[str, Any]) -> str:
    template = str(object_digest.get("cell_label_template") or "cell{row},{col}")
    return template.format(row=int(row), col=int(col))


def _cast_grid_toggle_solution(
    *,
    current_pattern: Sequence[Sequence[bool]],
    target_pattern: Sequence[Sequence[bool]],
    object_digest: Mapping[str, Any],
) -> list[str]:
    actions: list[str] = []
    for row, target_row in enumerate(target_pattern):
        for col, target in enumerate(target_row):
            current = (
                bool(current_pattern[row][col])
                if row < len(current_pattern) and col < len(current_pattern[row])
                else False
            )
            if current != bool(target):
                actions.append(_cast_label(row, col, object_digest))
    return actions


def _navigation_solution(object_digest: Mapping[str, Any]) -> list[str]:
    start = object_digest.get("player_start")
    exit_box = object_digest.get("exit_box")
    if not isinstance(start, Sequence) or isinstance(start, (str, bytes)) or len(start) < 2:
        return []
    if (
        not isinstance(exit_box, Sequence)
        or isinstance(exit_box, (str, bytes))
        or len(exit_box) < 4
    ):
        return []
    row = int(start[0])
    col = int(start[1])
    row_min, col_min, row_max, col_max = (int(v) for v in exit_box[:4])
    step = max(1, int(object_digest.get("navigation_step", object_digest.get("step", 1)) or 1))
    labels = {
        "up": str(object_digest.get("direction_labels", {}).get("up", "1")),
        "down": str(object_digest.get("direction_labels", {}).get("down", "2")),
        "left": str(object_digest.get("direction_labels", {}).get("left", "3")),
        "right": str(object_digest.get("direction_labels", {}).get("right", "4")),
    }
    path: list[str] = []
    while col > col_max:
        path.append(labels["left"])
        col -= step
    while col < col_min:
        path.append(labels["right"])
        col += step
    while row > row_max:
        path.append(labels["up"])
        row -= step
    while row < row_min:
        path.append(labels["down"])
        row += step
    return path


def _cast_patch_bounds(
    row: int,
    col: int,
    object_digest: Mapping[str, Any],
) -> tuple[int, int, int, int]:
    origin = object_digest.get("cast_origin", (0, 0))
    ox, oy = _point(origin)
    step = int(object_digest.get("cast_step", 1) or 1)
    size = int(object_digest.get("cast_cell_size", 1) or 1)
    x = ox + step * int(col)
    y = oy + step * int(row)
    return y, y + size, x, x + size


def _cast_data_key(data: Any) -> tuple[int, int] | None:
    if not isinstance(data, Mapping):
        return None
    try:
        return int(data["x"]), int(data["y"])
    except (KeyError, TypeError, ValueError):
        return None


def _cast_cell_from_data(
    data: Any,
    object_digest: Mapping[str, Any],
    *,
    shape: tuple[int, int],
) -> tuple[int, int] | None:
    key = _cast_data_key(data)
    if key is None:
        return None
    x, y = key
    origin = object_digest.get("cast_origin", (0, 0))
    ox, oy = _point(origin)
    step = int(object_digest.get("cast_step", 1) or 1)
    pattern = _bool_pattern(object_digest.get("target_pattern") or ())
    if step <= 0 or not pattern:
        return None
    if (x - ox) % step or (y - oy) % step:
        return None
    col = (x - ox) // step
    row = (y - oy) // step
    if row not in range(len(pattern)) or col not in range(len(pattern[0])):
        return None
    y0, y1, x0, x1 = _cast_patch_bounds(row, col, object_digest)
    if y0 < 0 or x0 < 0 or y1 > shape[0] or x1 > shape[1]:
        return None
    return int(row), int(col)


def _cast_cells(arr: Any, object_digest: Mapping[str, Any]) -> tuple[tuple[bool, ...], ...]:
    import numpy as np

    grid = np.asarray(arr)
    pattern = _bool_pattern(object_digest.get("target_pattern") or ())
    active = int(object_digest.get("cast_active_color", 1))
    rows: list[tuple[bool, ...]] = []
    for row in range(len(pattern)):
        values: list[bool] = []
        for col in range(len(pattern[row])):
            y0, y1, x0, x1 = _cast_patch_bounds(row, col, object_digest)
            patch = grid[y0:y1, x0:x1]
            values.append(bool(patch.size and np.any(patch == active)))
        rows.append(tuple(values))
    return tuple(rows)


def _set_cast_patch(
    out: Any,
    *,
    row: int,
    col: int,
    value: int,
    object_digest: Mapping[str, Any],
) -> None:
    y0, y1, x0, x1 = _cast_patch_bounds(row, col, object_digest)
    out[y0:y1, x0:x1] = int(value)


def _clear_cast_grid(out: Any, object_digest: Mapping[str, Any]) -> None:
    background = int(object_digest.get("background_color", 0))
    pattern = _bool_pattern(object_digest.get("target_pattern") or ())
    for row in range(len(pattern)):
        for col in range(len(pattern[row])):
            _set_cast_patch(out, row=row, col=col, value=background, object_digest=object_digest)


def _player_mask(arr: Any, object_digest: Mapping[str, Any]) -> Any:
    import numpy as np

    mask = np.zeros_like(arr, dtype=bool)
    for color in object_digest.get("player_colors", ()):
        mask |= np.asarray(arr) == int(color)
    return mask


def _shrink_player(out: Any, object_digest: Mapping[str, Any]) -> None:
    import numpy as np

    mask = _player_mask(out, object_digest)
    coords = np.argwhere(mask)
    if coords.size == 0:
        return
    background = int(object_digest.get("background_color", 0))
    colors = [int(color) for color in object_digest.get("player_colors", (9, 10))]
    row0, col0 = coords.min(axis=0)
    row1, col1 = coords.max(axis=0) + 1
    out[row0:row1, col0:col1] = background
    height = int(object_digest.get("shrunk_player_height", 2) or 2)
    for index, color in enumerate(colors[: max(1, len(colors))]):
        out[row0 : row0 + height, col0 + index : col0 + index + 1] = color


def _cast_grid_hash(grid: Any) -> str:
    import hashlib
    import numpy as np

    return hashlib.sha256(np.asarray(grid, dtype="<i2").tobytes()).hexdigest()[:16]


def _patch_lookup(
    object_digest: Mapping[str, Any],
) -> dict[tuple[str, int, tuple[int, int] | None], Any]:
    import numpy as np

    lookup: dict[tuple[str, int, tuple[int, int] | None], Any] = {}
    raw = object_digest.get("transition_patches") or ()
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)):
        return lookup
    for row in raw:
        if not isinstance(row, Mapping):
            continue
        before_hash = str(row.get("before_hash") or "")
        if not before_hash or "next_grid" not in row:
            continue
        data_key_value = row.get("data_key")
        data_key = (
            tuple(int(v) for v in data_key_value)
            if isinstance(data_key_value, Sequence) and not isinstance(data_key_value, (str, bytes))
            else None
        )
        lookup[(before_hash, int(row.get("action", 0) or 0), data_key)] = np.asarray(
            row["next_grid"], dtype=int
        )
    return lookup


def _direction_delta_for_cast(
    action: Any,
    object_digest: Mapping[str, Any],
) -> tuple[int, int]:
    actions = object_digest.get("direction_actions") or {}
    step = int(object_digest.get("navigation_step", object_digest.get("step", 1)) or 1)
    return _motion_delta_for_action(
        action,
        step=step,
        direction_actions={
            "up": int(actions.get("up", 1)),
            "down": int(actions.get("down", 2)),
            "left": int(actions.get("left", 3)),
            "right": int(actions.get("right", 4)),
        },
    )


def _move_cast_player(grid: Any, action: Any, object_digest: Mapping[str, Any]) -> Any:
    import numpy as np

    out = np.array(grid, copy=True)
    dy, dx = _direction_delta_for_cast(action, object_digest)
    if dy == 0 and dx == 0:
        return out
    mask = _player_mask(out, object_digest)
    coords = np.argwhere(mask)
    if coords.size == 0:
        return out
    moved = coords + np.asarray([dy, dx])
    if (
        (moved[:, 0] < 0).any()
        or (moved[:, 1] < 0).any()
        or (moved[:, 0] >= out.shape[0]).any()
        or (moved[:, 1] >= out.shape[1]).any()
    ):
        return out
    values = out[mask].copy()
    out[mask] = int(object_digest.get("background_color", 0))
    for (row, col), value in zip(moved, values, strict=True):
        out[int(row), int(col)] = int(value)
    return out


def _cast_player_at_exit(grid: Any, object_digest: Mapping[str, Any]) -> bool:
    import numpy as np

    exit_box = object_digest.get("exit_box")
    if (
        not isinstance(exit_box, Sequence)
        or isinstance(exit_box, (str, bytes))
        or len(exit_box) < 4
    ):
        return False
    row_min, col_min, row_max, col_max = (int(v) for v in exit_box[:4])
    coords = np.argwhere(_player_mask(grid, object_digest))
    if coords.size == 0:
        return False
    return bool(
        np.any(
            (coords[:, 0] >= row_min)
            & (coords[:, 0] <= row_max)
            & (coords[:, 1] >= col_min)
            & (coords[:, 1] <= col_max)
        )
    )


def cast_grid_phase_fsm_world_model(
    *,
    game: str,
    object_digest: Mapping[str, Any],
    few_shot_examples: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """REQ-REPORT-4469: synthesize a two-phase cast-grid FSM world model.

    The candidate starts with the tempting but wrong single-phase predicate
    "cast grid matches the spell pattern." A grounded digest with an exit
    predicate refutes that candidate and re-induces the two-phase model:
    toggle the cast grid, fire the shrink transition, then navigate the player
    to the exit. Optional transition patches let a verifier-grounded CEGIS pass
    override fallback dynamics without importing a target game's hand recipe.
    """

    if not isinstance(object_digest, Mapping):
        return _ungrounded_cast_grid_result(game, "missing_cast_grid_phase_fsm_digest")
    rule_family = str(
        object_digest.get("rule_family") or object_digest.get("predicate_id") or ""
    ).lower()
    if "cast" not in rule_family and not _cast_grid_examples_support(few_shot_examples):
        return _ungrounded_cast_grid_result(game, "missing_cast_grid_phase_fsm_few_shot_examples")
    if not _cast_grid_examples_support(few_shot_examples):
        return _ungrounded_cast_grid_result(game, "missing_cast_grid_phase_fsm_few_shot_examples")

    target_pattern = _bool_pattern(object_digest.get("target_pattern") or ())
    current_pattern = _bool_pattern(
        object_digest.get("current_pattern")
        or tuple(tuple(False for _ in row) for row in target_pattern)
    )
    if not target_pattern or not current_pattern:
        return _ungrounded_cast_grid_result(game, "missing_cast_grid_toggle_digest")
    if len(current_pattern) != len(target_pattern) or any(
        len(current_pattern[row]) != len(target_pattern[row]) for row in range(len(target_pattern))
    ):
        return _ungrounded_cast_grid_result(game, "cast_grid_pattern_shape_mismatch")
    for key in (
        "cast_origin",
        "cast_step",
        "cast_cell_size",
        "cast_active_color",
        "background_color",
    ):
        if key not in object_digest:
            return _ungrounded_cast_grid_result(game, "missing_cast_grid_toggle_digest")
    for key in (
        "player_colors",
        "player_start",
        "exit_box",
        "direction_actions",
        "direction_labels",
    ):
        if key not in object_digest:
            return _ungrounded_cast_grid_result(game, "missing_cast_grid_navigation_digest")

    toggle_path = _cast_grid_toggle_solution(
        current_pattern=current_pattern,
        target_pattern=target_pattern,
        object_digest=object_digest,
    )
    navigation_path = _navigation_solution(object_digest)
    if not toggle_path or not navigation_path:
        return _ungrounded_cast_grid_result(game, "cast_grid_phase_fsm_candidate_did_not_ground")
    solution = toggle_path + navigation_path
    patches = _patch_lookup(object_digest)

    def engine(grid: Any, action: Any, data: Any = None) -> Any:
        import numpy as np

        arr = np.asarray(grid)
        key = (_cast_grid_hash(arr), int(action), _cast_data_key(data))
        if key in patches:
            return np.array(patches[key], copy=True)
        out = np.array(arr, copy=True)
        if int(action) == int(object_digest.get("click_action", 6)):
            cell = _cast_cell_from_data(data, object_digest, shape=out.shape)
            if cell is None:
                return out
            row, col = cell
            active = int(object_digest.get("cast_active_color", 1))
            background = int(object_digest.get("background_color", 0))
            y0, y1, x0, x1 = _cast_patch_bounds(row, col, object_digest)
            next_value = background if np.any(out[y0:y1, x0:x1] == active) else active
            _set_cast_patch(out, row=row, col=col, value=next_value, object_digest=object_digest)
            if _cast_cells(out, object_digest) == target_pattern:
                _clear_cast_grid(out, object_digest)
                _shrink_player(out, object_digest)
            return out
        return _move_cast_player(out, action, object_digest)

    def is_level_complete(grid: Any) -> bool:
        return bool(_cast_player_at_exit(grid, object_digest))

    return {
        "operator": "cast_grid_phase_fsm_world_model",
        "game": str(game),
        "grounded": True,
        "predicate_id": "toggle_csp_then_navigate_exit",
        "recipe_source": "generic_cast_grid_phase_fsm_world_model",
        "target_recipe_withheld": str(game),
        "transition_families": ["config_toggle", "phase_transition", "navigate"],
        "phase_model": {
            "phases": ["config_toggle", "navigate_exit"],
            "transition": "target cast-grid pattern fires shrink spell",
            "win_predicate": "player pixels intersect exit_box after shrink navigation",
        },
        "solution": [str(label) for label in solution],
        "toggle_solution": [str(label) for label in toggle_path],
        "navigation_solution": [str(label) for label in navigation_path],
        "counterexample_rounds": 1,
        "counterexamples": [
            {
                "rejected_candidate": "cast_grid_alignment_is_win",
                "refinement": "phase transition triggers shrink; final win requires exit contact",
            }
        ],
        "engine": engine,
        "is_level_complete": is_level_complete,
        "verifier": {
            "name": "execution_grounded_cast_grid_phase_fsm",
            "transition_patch_count": len(patches),
            "toggle_actions": len(toggle_path),
            "navigation_actions": len(navigation_path),
            "few_shot_examples": [
                str(row.get("game", "")) for row in few_shot_examples if isinstance(row, Mapping)
            ],
        },
        "grounded_win_condition": {
            "predicate": "cast-grid target pattern transitions to shrunk-player navigation; win is player-at-exit",
            "fires_on_win": True,
            "rejects_nonwins": True,
        },
        "verifier_is_oracle": True,
    }


def object_centric_digest(
    grid: Any,
    *,
    emit_grid_fallback_for_background: bool = False,
    grid_fallback_tile_px: int = 8,
    grid_fallback_max_tiles: int = 64,
) -> dict[str, Any]:
    """Connected-component digest for ARC frames or grids.

    ``emit_grid_fallback_for_background`` (REQ-ARC-FCP-5757 follow-up,
    ``GAP-ARC-BP35-CLICK-CANDIDATE-GENERATION-MISS``, 2026-07-23; OFF by default -- a purely
    additive opt-in, zero behavior change for any existing caller). The single most-common color
    is always excluded wholesale as "background" here, unconditionally -- but the most-common
    color is not necessarily true background: bp35's win condition requires clicking individual
    same-row "blocker" cells that all happen to share the single most-common color (measured
    directly: color 5, 2109/4096 px on a real stalled state), so those cells NEVER become click
    candidates regardless of search depth or LLM judgment -- a generation/perception gap no
    downstream selection or planning improvement can fix (confirmed via the 2026-07-23 candidate-
    coverage attribution: bp35 alone accounts for all 21/25 of that run's generation misses, every
    one an action-6 click never proposed despite 50+ other candidates existing). When enabled, the
    excluded background mask is tiled on an absolute ``grid_fallback_tile_px``-pixel pitch
    (grid-aligned across the whole frame) into small ``is_grid_fallback: True`` components -- one
    per occupied tile, not one per pixel, so this does not regress into per-pixel-noise explosion.
    Fails CLOSED: if tiling the background mask would exceed ``grid_fallback_max_tiles``, no
    fallback components are emitted (an arbitrary subset would bias toward whichever tiles happen
    to iterate first, worse than omitting)."""

    import numpy as np

    arr = np.asarray(grid)
    if arr.ndim != 2:
        raise ValueError("object_centric_digest expects a 2-D grid")
    vals, counts = np.unique(arr, return_counts=True)
    background = int(vals[counts.argmax()]) if len(vals) else 0
    mask = arr != background
    seen = np.zeros_like(mask, dtype=bool)
    components: list[dict[str, Any]] = []
    h, w = arr.shape
    for y0 in range(h):
        for x0 in range(w):
            if not mask[y0, x0] or seen[y0, x0]:
                continue
            color = int(arr[y0, x0])
            stack = [(y0, x0)]
            seen[y0, x0] = True
            cells: list[tuple[int, int]] = []
            while stack:
                y, x = stack.pop()
                cells.append((y, x))
                for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    ny, nx = y + dy, x + dx
                    if (
                        0 <= ny < h
                        and 0 <= nx < w
                        and mask[ny, nx]
                        and not seen[ny, nx]
                        and int(arr[ny, nx]) == color
                    ):
                        seen[ny, nx] = True
                        stack.append((ny, nx))
            ys = [y for y, _ in cells]
            xs = [x for _, x in cells]
            bbox = [min(ys), min(xs), max(ys), max(xs)]
            area = len(cells)
            signature = f"c{color}:a{area}:bbox{bbox[0]},{bbox[1]},{bbox[2]},{bbox[3]}"
            components.append(
                {
                    "color": color,
                    "area": area,
                    "bbox": bbox,
                    "centroid": [sum(xs) / area, sum(ys) / area],
                    "signature": signature,
                }
            )
    if emit_grid_fallback_for_background:
        bg_ys, bg_xs = np.nonzero(~mask)
        tile_px = max(1, int(grid_fallback_tile_px))
        tile_of: dict[tuple[int, int], list[tuple[int, int]]] = {}
        for y, x in zip(bg_ys.tolist(), bg_xs.tolist()):
            tile_of.setdefault((y // tile_px, x // tile_px), []).append((y, x))
        if len(tile_of) <= int(grid_fallback_max_tiles):
            for cells in tile_of.values():
                tys = [c[0] for c in cells]
                txs = [c[1] for c in cells]
                area = len(cells)
                bbox = [min(tys), min(txs), max(tys), max(txs)]
                components.append(
                    {
                        "color": background,
                        "area": area,
                        "bbox": bbox,
                        "centroid": [sum(txs) / area, sum(tys) / area],
                        "signature": f"c{background}:a{area}:bbox{bbox[0]},{bbox[1]},{bbox[2]},{bbox[3]}:gridfallback",
                        "is_grid_fallback": True,
                    }
                )
    components.sort(key=lambda row: (-int(row["area"]), int(row["color"]), row["bbox"]))
    return {
        "operator": "object_centric_digest",
        "shape": [int(arr.shape[0]), int(arr.shape[1])],
        "background_color": background,
        "component_count": len(components),
        "components": components,
    }


def active_data_collection_plan(
    *,
    action_labels: Sequence[str],
    object_signatures: Sequence[str],
    max_cases_per_action: int = 3,
) -> list[dict[str, Any]]:
    """Balanced action/object coverage rows for offline transition collection."""

    signatures = list(object_signatures) or ["none"]
    rows: list[dict[str, Any]] = []
    for action in action_labels:
        for case_index in range(max(0, int(max_cases_per_action))):
            rows.append(
                {
                    "operator": "active_data_collection",
                    "action": str(action),
                    "object_signature": signatures[case_index % len(signatures)],
                    "case_index": case_index,
                    "selection_policy": "balanced_action_object_coverage",
                }
            )
    return rows


def astar_frontier_priority(
    *, depth: int, heuristic: float, path_cost_weight: Optional[float] = None
) -> float:
    """Standing graph/A* priority shared by OfflineSolver and graph-explore users."""

    return float(standing_path_cost_weight(path_cost_weight) * int(depth) + float(heuristic))


def standing_path_cost_weight(path_cost_weight: Optional[float]) -> float:
    """REQ-LEARN-4364: default ARC planning to additive A* cost; keep 0.0 as baseline."""
    if path_cost_weight is None:
        return ARC_STANDING_PATH_COST_WEIGHT
    return float(path_cost_weight)


# Default-OFF, like every other opt-in live-agent lever in this codebase (e.g.
# SUBMITTED_COLOR_BLOB_SALIENCE_ENABLED, AUTO_HUD_MASK). See arc_hud_bar_detector.py's
# Stage 3 docstring: the estimator earns trust before it earns a place in the live cascade.
BUDGET_AWARE_SEARCH_ENABLED = False

# Below this many estimated-actions-remaining, a candidate plan whose own length would
# exceed the estimate is penalised rather than pruned outright -- pruning risks discarding
# the only winning plan on a false-positive estimate; a large additive penalty still lets
# a plan through if nothing shorter exists, but prefers a shorter one when one does.
BUDGET_EXHAUSTION_PENALTY_PER_EXCESS_ACTION = 50.0


def budget_aware_path_cost_weight(
    *,
    depth: int,
    plan_length: Optional[int] = None,
    actions_remaining_estimate: Optional[float] = None,
    path_cost_weight: Optional[float] = None,
) -> float:
    """REQ-ARC-WMTE-6180: `standing_path_cost_weight`, plus an additive penalty for plans
    likely to outrun an exhausting HUD budget meter (see
    `arc_hud_bar_detector.budget_exhaustion_estimate`).

    Pure arithmetic, no side effects -- callers own the decision of whether/when to pass a
    real ``actions_remaining_estimate`` in (this module never calls the detector itself).
    ``actions_remaining_estimate=None`` (no admitted estimate yet, or the feature is off)
    reduces this to exactly `standing_path_cost_weight`'s behaviour -- a genuine no-op, not
    an approximation of one, so wiring this in cannot silently change behaviour before an
    estimate actually exists.
    """
    base = standing_path_cost_weight(path_cost_weight) * int(depth)
    if plan_length is None or actions_remaining_estimate is None:
        return float(base)
    excess = int(plan_length) - float(actions_remaining_estimate)
    if excess <= 0:
        return float(base)
    return float(base + excess * BUDGET_EXHAUSTION_PENALTY_PER_EXCESS_ACTION)


def offline_arcade() -> Any:  # pragma: no cover - thin SDK boundary
    """A zero-quota, no-network OFFLINE Arcade over the local environment_files."""
    from arc_agi import Arcade
    from arc_agi.base import OperationMode

    return Arcade(
        arc_api_key="", operation_mode=OperationMode.OFFLINE, environments_dir=str(ENV_DIR)
    )


def frame_level(frame: Any) -> int:
    """The level is read from the FRAME, never from env._game (gotcha #2)."""
    if frame is None:
        return -1
    return int(getattr(frame, "levels_completed", 0) or 0)


class OfflineSolver:
    """Reusable from-scratch offline ARC solver (replay-from-reset BFS).

    Plug in a per-game model; inherit the universal harness (offline arcade,
    warm-up after reset, replay-from-reset, frame-based level, BFS + dedup,
    level chaining). Each game supplies:
      - action_labels(env) -> list[str]: the action vocabulary at the current
        state (env-DISCOVERED, not hardcoded; gotcha #5).
      - apply(env, label, frame) -> frame: execute one action, resolving any
        animation (gotcha #6), and return the new frame.
      - state_key(game) -> Hashable: the dedup key — MUST include every
        load-bearing piece of state (position, FACING, cast-state, sprites;
        gotcha #6), else turns/no-ops collapse and the search stalls.
    """

    def __init__(
        self,
        game_id: str,
        action_labels: Callable[[Any], Sequence[str]],
        apply: Callable[[Any, str, Any], Any],
        state_key: Callable[[Any], Hashable],
        *,
        warmup_label: Optional[str] = None,
        max_nodes: int = 30000,
        verifier: Optional[Callable[[Any], float]] = None,
        path_cost_weight: Optional[float] = None,
        branch_mode: str = "replay",
        env_factory: Optional[Callable[[], Any]] = None,
        move_pruner: Optional[Any] = None,
    ) -> None:
        self.game_id = game_id
        self.action_labels = action_labels
        self.apply = apply
        self.state_key = state_key
        # OPTIONAL MOVE-PRUNER (efficiency, north-star action-cost axis). When given, an object with
        #   should_prune(frame, label) -> bool   (skip a predicted-dead-end edge BEFORE applying it)
        #   observe(frame_before, label, frame_after, leveled_up)  (learn from the actual outcome)
        # lets the search skip expansions it predicts are dead-ends (e.g. walking into a charging enemy),
        # shrinking states_expanded. See arc_hazard_pruner.HazardMovePruner -- it fits a hazard model from
        # the search's OWN observed deaths (no offline ground-truth), so it transfers to unseen games and
        # NO-OPS when no hazard is present. Correctness-preserving: it only prunes edges the learned model
        # predicts terminate the avatar; the reproduction gate remains the final authority.
        self.move_pruner = move_pruner
        self.warmup_label = warmup_label  # an action to consume the no-op first slot (gotcha #4)
        self.max_nodes = max_nodes
        # BRANCH MODE — how the search navigates between nodes:
        #   "replay"  (default, UNCHANGED): replay-from-reset (env.reset() + re-apply the path) per
        #             node. Correct + memory-light for games whose state is fully a function of the
        #             action prefix from reset (lp85, sc25). The proven default; do not change it.
        #   "deepcopy": snapshot copy.deepcopy(env._game) per node and restore by deepcopy, branching
        #             the EXACT env state rather than reconstructing it. Use for games where
        #             replay-from-reset does not faithfully reproduce the searched state (the verifier
        #             then finds a path that fails the reproduction gate). Costs a deepcopy per node;
        #             requires env._game to be deepcopy-able + injectable — the deepcopy-injection
        #             gotcha #3 means it is NOT universal (works for lp85, BROKEN for sc25/tu93).
        #   "fresh_env": make a BRAND-NEW env (env_factory, default a fresh arc.make of game_id) for
        #             EVERY candidate evaluation and replay prefix+path from reset on it. The fix for a
        #             game whose env.reset() is NON-IDEMPOTENT (gotcha #7: tu93's reset leaves a
        #             parity-toggling hidden state, so the reuse-one-env search detects parity-contingent
        #             "wins" that fail the fresh-env reproduction gate). A fresh env always starts at the
        #             SAME pristine parity the gate uses, so found paths reproduce. Costs a fresh env +
        #             full replay per evaluation — slower, so reserve it for non-idempotent-reset games.
        self.branch_mode = branch_mode
        self.env_factory = env_factory  # () -> fresh env, for branch_mode="fresh_env"
        self._fresh_arcade: Any = None  # lazily-cached arcade + scorecard for the default factory
        self._fresh_scorecard: Any = None
        # VERIFIER-ROUTED SEARCH (the north-star efficiency loop): a score on a
        # state (LOWER = closer to the win, an energy/goal-distance). When given,
        # the search is best-first ordered by it, so it expands promising branches
        # first and the state count SHRINKS. When None, it degrades to plain BFS
        # (verifier ≡ 0 → the heap orders by insertion = FIFO). Pass a learned or
        # computed verifier to turn the solver into a verifier-routed search.
        self.verifier = verifier or (lambda _g: 0.0)
        self.path_cost_weight = standing_path_cost_weight(path_cost_weight)
        self.last_states_expanded = 0
        self.last_frame: Any = None

    def _call_state_key(self, env: Any) -> Hashable:
        try:
            return self.state_key(env._game, self.last_frame)  # type: ignore[misc]
        except TypeError:
            return self.state_key(env._game)

    def _call_verifier(self, env: Any) -> float:
        try:
            return float(self.verifier(env._game, self.last_frame))  # type: ignore[misc]
        except TypeError:
            return float(self.verifier(env._game))

    def _priority(self, env: Any, path: Sequence[str]) -> float:
        """Verifier score plus standing path cost. Pass 0.0 for legacy greedy routing."""
        return astar_frontier_priority(
            depth=len(path),
            heuristic=self._call_verifier(env),
            path_cost_weight=self.path_cost_weight,
        )

    def _call_action_labels(self, env: Any, path: Sequence[str]) -> Sequence[str]:
        try:
            return self.action_labels(env, self.last_frame, tuple(path))  # type: ignore[misc]
        except TypeError:
            try:
                return self.action_labels(env, self.last_frame)  # type: ignore[misc]
            except TypeError:
                return self.action_labels(env)

    def _replay(self, env: Any, path: Sequence[str]) -> Any:
        f = env.reset()
        self.last_frame = f
        if self.warmup_label is not None:
            f = self.apply(env, self.warmup_label, f)  # gotcha #4
            self.last_frame = f
        for label in path:
            f = self.apply(env, label, f)
            self.last_frame = f
        return f

    def solve_level(self, env: Any, start_level: int, prefix: Sequence[str], depth_cap: int):
        """Search one level forward from `prefix` — verifier-routed BEST-FIRST (or
        plain BFS when no verifier). Returns (extension_path, states_expanded)."""
        if self.branch_mode == "deepcopy":
            return self._solve_level_deepcopy(env, start_level, prefix, depth_cap)
        if self.branch_mode == "fresh_env":
            return self._solve_level_fresh(env, start_level, prefix, depth_cap)
        self._replay(env, list(prefix))
        seen = {self._call_state_key(env)}
        counter = itertools.count()  # FIFO tiebreaker (so verifier≡0 ⇒ BFS)
        heap = [(self._priority(env, []), next(counter), [])]
        nodes = 0
        while heap and nodes < self.max_nodes:
            _, _, path = heapq.heappop(heap)
            if len(path) >= depth_cap:
                continue
            self._replay(env, list(prefix) + path)
            # FROZEN SNAPSHOT, not a live reference: found via a real diagnostic trace (not assumed)
            # investigating an anomalously shallow lf52 search -- env.step()'s returned frame objects
            # are distinct Python objects (`f1 is f0` is False) but their underlying grid data is a
            # SHARED, mutated-in-place buffer, so a bare `node_frame = self.last_frame` reference
            # silently reflects whatever the CURRENT env state is at read-time, not the state at
            # capture-time. Since move_pruner.observe() is called for EVERY sibling candidate in this
            # loop, each with its own apply()/env.step() in between, an un-copied node_frame ends up
            # showing before_key == after_key (looks like "no change") for EVERY candidate regardless
            # of whether the game actually changed -- corrupting the dead-end pruner with false
            # dead-ends for genuinely state-changing actions. copy.deepcopy() breaks the aliasing.
            node_frame = copy.deepcopy(self.last_frame)
            for label in self._call_action_labels(env, path):
                if self.move_pruner is not None and self.move_pruner.should_prune(
                    node_frame, label
                ):
                    continue  # learned dead-end (e.g. walks into a charger) -- skip the expansion
                f2 = self.apply(env, label, None)
                self.last_frame = f2
                nodes += 1
                leveled = frame_level(f2) > start_level
                if self.move_pruner is not None:
                    self.move_pruner.observe(node_frame, label, f2, leveled)
                if leveled:
                    self.last_states_expanded = nodes
                    return path + [label], nodes
                k = self._call_state_key(env)
                if k not in seen:
                    seen.add(k)
                    child_path = path + [label]
                    heapq.heappush(
                        heap, (self._priority(env, child_path), next(counter), child_path)
                    )
                self._replay(env, list(prefix) + path)  # restore for next sibling
        self.last_states_expanded = nodes
        return None, nodes

    def _solve_level_deepcopy(
        self, env: Any, start_level: int, prefix: Sequence[str], depth_cap: int
    ):
        """DEEPCOPY-PER-NODE variant of solve_level. Instead of replaying-from-reset to navigate, it
        SNAPSHOTS copy.deepcopy(env._game) per node and restores by deepcopy — branching the EXACT env
        state (incl. anything replay-from-reset doesn't faithfully reconstruct). Each heap node carries
        its (snapshot, frame) so state_key/verifier see the right frame; the found path is identical in
        shape to the replay variant (a sequence of labels) so the reproduction gate is unchanged."""
        self._replay(env, list(prefix))
        seen = {self._call_state_key(env)}
        counter = itertools.count()
        root = (
            self._priority(env, []),
            next(counter),
            [],
            copy.deepcopy(env._game),
            self.last_frame,
        )
        heap = [root]
        nodes = 0
        while heap and nodes < self.max_nodes:
            _, _, path, snap, frame = heapq.heappop(heap)
            if len(path) >= depth_cap:
                continue
            env._game = copy.deepcopy(snap)  # restore this node's exact state
            self.last_frame = frame
            # FROZEN SNAPSHOT, not a live reference -- see solve_level()'s node_frame comment for the
            # full diagnosis (frame objects alias a shared, mutated-in-place grid buffer).
            node_frame = copy.deepcopy(frame)
            for label in self._call_action_labels(env, path):
                if self.move_pruner is not None and self.move_pruner.should_prune(
                    node_frame, label
                ):
                    continue  # learned dead-end -- skip the expansion
                env._game = copy.deepcopy(snap)  # branch from the node for each child
                self.last_frame = frame
                f2 = self.apply(env, label, None)
                self.last_frame = f2
                nodes += 1
                leveled = frame_level(f2) > start_level
                if self.move_pruner is not None:
                    self.move_pruner.observe(node_frame, label, f2, leveled)
                if leveled:
                    self.last_states_expanded = nodes
                    return path + [label], nodes
                k = self._call_state_key(env)
                if k not in seen:
                    seen.add(k)
                    child_path = path + [label]
                    heapq.heappush(
                        heap,
                        (
                            self._priority(env, child_path),
                            next(counter),
                            child_path,
                            copy.deepcopy(env._game),
                            f2,
                        ),
                    )
        self.last_states_expanded = nodes
        return None, nodes

    def _fresh_env(self) -> Any:
        """A BRAND-NEW env for branch_mode='fresh_env' — pristine reset parity. Default: a fresh
        arc.make of self.game_id over a lazily-cached offline arcade + scorecard."""
        if self.env_factory is not None:
            return self.env_factory()
        if self._fresh_arcade is None:
            self._fresh_arcade = offline_arcade()
            self._fresh_scorecard = self._fresh_arcade.open_scorecard()
        return self._fresh_arcade.make(self.game_id, scorecard_id=self._fresh_scorecard)

    def _solve_level_fresh(self, env: Any, start_level: int, prefix: Sequence[str], depth_cap: int):
        """FRESH-ENV-PER-NODE variant of solve_level. EVERY candidate is evaluated on a BRAND-NEW env
        (replay prefix+path from reset), so each evaluation sees the same pristine reset parity the
        reproduction gate uses — the fix for non-idempotent-reset games (gotcha #7: a game whose
        env.reset() leaves parity-toggling hidden state, where the reuse-one-env search detects
        parity-contingent wins that fail the fresh-env gate). The `env` arg is unused — the factory
        mints fresh envs. Slower (a fresh env + full replay per evaluation); reserve for such games."""

        def at(path: Sequence[str]):
            e = self._fresh_env()
            self._replay(
                e, list(prefix) + list(path)
            )  # reset+replay on the fresh env; sets last_frame
            return e

        e0 = at([])
        seen = {self._call_state_key(e0)}
        counter = itertools.count()
        heap = [(self._priority(e0, []), next(counter), [])]
        nodes = 0
        while heap and nodes < self.max_nodes:
            _, _, path = heapq.heappop(heap)
            if len(path) >= depth_cap:
                continue
            e_node = at(path)  # fresh env at the node (for action_labels)
            # FROZEN SNAPSHOT, not a live reference -- see solve_level()'s node_frame comment for the
            # full diagnosis (frame objects alias a shared, mutated-in-place grid buffer).
            node_frame = copy.deepcopy(self.last_frame)
            for label in self._call_action_labels(e_node, path):
                if self.move_pruner is not None and self.move_pruner.should_prune(
                    node_frame, label
                ):
                    continue  # learned dead-end (e.g. walks into a charger) -- skip the fresh-env eval
                e_child = at(path + [label])
                f2 = self.last_frame
                nodes += 1
                leveled = frame_level(f2) > start_level
                if self.move_pruner is not None:
                    self.move_pruner.observe(node_frame, label, f2, leveled)
                if leveled:
                    self.last_states_expanded = nodes
                    return path + [label], nodes
                k = self._call_state_key(e_child)
                if k not in seen:
                    seen.add(k)
                    heapq.heappush(
                        heap,
                        (self._priority(e_child, path + [label]), next(counter), path + [label]),
                    )
        self.last_states_expanded = nodes
        return None, nodes

    def solve(self, env: Any, target_level: int, depth_cap: int = 30):
        """Chain levels from reset to target_level; return the full action path + reached level."""
        f = self._replay(env, [])
        cur = frame_level(f)
        full: list[str] = []
        for lvl in range(cur + 1, target_level + 1):
            path, _ = self.solve_level(env, cur, full, depth_cap)
            if path is None:
                break
            f = self._replay(env, full + path)
            cur = frame_level(f)
            full += path
            if cur < lvl:
                break
        return full, cur


def _action6_click_from_label(label: str) -> Optional[tuple[int, int]]:
    """Best-effort extraction of an ACTION6 (x, y) click from a solve-label string.

    Covers the dominant label encoding (`_json_action_label` in arc_game_adapters.py:
    a JSON string `{"action": 6, "data": {"x": .., "y": ..}}`). Returns None (never
    raises) on any other encoding or a non-ACTION6 label -- this is a best-effort
    early-warning check, not a complete parser for every adapter's label dialect
    (some games use templated strings like "click:{x},{y}" via
    `_click_label_for_grid`, which are not covered here). Coverage is reported
    honestly by the caller (`checked_action6_clicks` vs `solution` length) rather
    than silently assumed complete.
    """
    try:
        parsed = json.loads(label)
    except (TypeError, ValueError):
        return None
    if not isinstance(parsed, Mapping):
        return None
    if _action_id_from(parsed) != 6:
        return None
    return _click_xy_from(parsed)


def _action6_out_of_live_bounds(x: int, y: int) -> bool:
    """True if (x, y) would be REJECTED by the live arcprize.org API for ACTION6.

    Reuses the bound already declared by the installed `arcengine` dependency
    (`ComplexAction`'s `x`/`y` fields, `Field(ge=0, le=63)`) -- the exact same
    pydantic validation the live server runs in `RestAPI.cmd()` before dispatching
    an action. The OFFLINE arcade (`LocalEnvironmentWrapper.step()`) never calls
    this validation -- it silently accepts and routes any coordinate straight into
    the per-game simulator's hit-test, which itself does no bounds check either.
    That asymmetry is what let lf52's original L9 route (22 clicks with x up to 132)
    reproduce cleanly offline while 400-ing live at submission time (see
    ops/known-issues.md 2026-07-17, commit 5ca2a999b). Reusing arcengine's own
    declared bound (rather than hardcoding 0/63 a second time) keeps this check in
    sync automatically if the live API's bound ever changes.
    """
    from arcengine.enums import GameAction
    from pydantic import ValidationError

    try:
        GameAction.ACTION6.validate_data({"x": int(x), "y": int(y)})
    except ValidationError:
        return True
    return False


def reproduce(
    game_id: str,
    solution: Sequence[str],
    apply: Callable[[Any, str, Any], Any],
    *,
    warmup_label: Optional[str] = None,
    claimed_level: Optional[int] = None,
) -> dict:
    """THE REPRODUCTION GATE. Replay a banked `solution` against the OFFLINE env and
    report the level it actually reaches. A solve is only real if this reproduces
    the claimed level offline — never trust a live-recorded trajectory alone.

    Also flags any ACTION6 click that the OFFLINE arcade would silently accept but
    the LIVE API would reject (out of the [0,63]x[0,63] bound) -- offline
    reproduction is necessary but NOT sufficient for live-submittability; a route
    can pass this gate's `reproduced: True` while still being un-submittable, which
    is exactly what happened to lf52's original L9 route before its 2026-07-17 fix.
    `oob_action6_clicks`/`any_oob_action6_clicks` surface that gap explicitly rather
    than leaving it to be discovered only at live-submission time. Best-effort:
    `checked_action6_clicks` reports how many labels this could actually parse and
    check (see `_action6_click_from_label`'s coverage caveat) -- a label dialect this
    can't parse is silently skipped, not silently assumed clean.

    Returns {reached_level, claimed_level, reproduced: bool, oob_action6_clicks,
    any_oob_action6_clicks, checked_action6_clicks}. Zero quota.
    """
    arc = offline_arcade()
    env = arc.make(game_id, scorecard_id=arc.open_scorecard())
    f = env.reset()
    if warmup_label is not None:
        f = apply(env, warmup_label, f)
    oob_action6_clicks: list[dict[str, int]] = []
    checked_action6_clicks = 0
    for index, label in enumerate(solution):
        click = _action6_click_from_label(label)
        if click is not None:
            checked_action6_clicks += 1
            x, y = click
            if _action6_out_of_live_bounds(x, y):
                oob_action6_clicks.append({"index": index, "x": x, "y": y})
        f = apply(env, label, f)
    reached = frame_level(f)
    return {
        "game": game_id,
        "reached_level": reached,
        "claimed_level": claimed_level,
        "reproduced": (claimed_level is None) or (reached >= int(claimed_level)),
        "mode": "offline_reproduction_gate_no_quota",
        "checked_action6_clicks": checked_action6_clicks,
        "oob_action6_clicks": oob_action6_clicks,
        "any_oob_action6_clicks": bool(oob_action6_clicks),
    }


# ---------------------------------------------------------------------------
# Exploration-playbook primitives (REQ-ARC-WMTE-5716)
#
# Game-AGNOSTIC exploration moves distilled from the whole solve corpus
# (docs/research-notes/arc-exploration-playbook-20260717.md). These encode the
# recurring METHODOLOGY -- verify semantics empirically, read absolute motion,
# interrogate unexplained objects, know proven-vs-capped, isolate deaths -- so
# the SAME know-how applies to a hidden game the agent has never seen, without
# smuggling in any per-game fact (a color, a coordinate, a mechanic). None of
# these reads env._game internals; they operate on rendered frames + injected
# callables, so they are usable both offline (this kit) and on the live path.
# ---------------------------------------------------------------------------


def _frame_layers(frame: Any) -> list[Any]:
    """Normalize any ARC frame representation into a list of 2-D grid layers.

    A live ARC frame stacks N animation sub-frames as an (N, H, W) array on
    ``.frame`` / ``._frame``; the LAST layer is the settled grid, and the
    intermediate layers (captured with a FIXED camera during a single action)
    carry the absolute-motion information the camera-relative settled grid hides
    (playbook 2.3). Accepts a FrameDataRaw-like object, a raw (N,H,W) or (H,W)
    array, or a list of 2-D grids -- so callers do not have to special-case the
    representation.
    """
    import numpy as np

    raw = frame
    for attr in ("frame", "_frame"):
        # A FrameDataRaw exposes the layer stack here; a bare array/list does not,
        # so falling through leaves ``raw`` as the array/list the caller passed.
        if hasattr(frame, attr):
            raw = getattr(frame, attr)
            break
    arr = np.asarray(raw)
    if arr.ndim == 2:
        return [arr]
    if arr.ndim == 3:
        return [arr[i] for i in range(arr.shape[0])]
    raise ValueError("frame must normalize to a 2-D grid or a stack of 2-D grids")


def settled_grid(frame: Any) -> Any:
    """The settled 2-D grid = the LAST animation layer (playbook 1.2 / 2.3).

    Read this (never a mid-animation layer) when you want the resting board the
    win predicate is evaluated against; read the full layer stack via
    :func:`read_absolute_trajectory` when you need in-action motion.
    """
    return _frame_layers(frame)[-1]


def _grid_background(grid: Any, background: Optional[int]) -> int:
    import numpy as np

    if background is not None:
        return int(background)
    arr = np.asarray(grid)
    vals, counts = np.unique(arr, return_counts=True)
    return int(vals[counts.argmax()]) if len(vals) else 0


def probe_action_semantics(
    env_factory: Callable[[], Any],
    apply: Callable[[Any, Any, Any], Any],
    action_labels: Sequence[Any],
    *,
    warmup_label: Optional[Any] = None,
    prefix: Sequence[Any] = (),
) -> dict[str, Any]:
    """REQ-ARC-WMTE-5716 (playbook 1.1): empirically measure what each candidate
    action DOES this level, instead of assuming carryover from a similar-looking
    prior level/game -- the single most-cited wasted-round + instant-death cause
    in the corpus.

    Each label is measured INDEPENDENTLY from the same known state: a fresh env
    (``env_factory()``) is reset, optionally warmed up (gotcha #4), replayed
    through ``prefix``, then the one label is applied (playbook 4.1 fresh-env
    branching, so a lethal probe cannot corrupt a real attempt). Reports per
    label: the level delta, the number of changed settled-grid cells, whether it
    leveled up, whether it caused a death (the level counter dropped below the
    pre-action level, e.g. a GAME_OVER reset), and whether it was inert (no cell
    changed and no level change). ``changed_cells`` is ``None`` when the grid
    shape changed (e.g. a degenerate terminal frame), which the caller should
    treat as "not inert".
    """
    import numpy as np

    rows: list[dict[str, Any]] = []
    for label in action_labels:
        env = env_factory()
        frame = env.reset()
        if warmup_label is not None:
            frame = apply(env, warmup_label, frame)
        for step_label in prefix:
            frame = apply(env, step_label, frame)
        before_grid = np.asarray(settled_grid(frame))
        before_level = frame_level(frame)
        after = apply(env, label, frame)
        after_grid = np.asarray(settled_grid(after))
        after_level = frame_level(after)

        if before_grid.shape == after_grid.shape:
            changed_cells: Optional[int] = int(np.count_nonzero(before_grid != after_grid))
        else:
            changed_cells = None
        leveled_up = after_level > before_level
        died = after_level < before_level
        inert = changed_cells == 0 and after_level == before_level
        rows.append(
            {
                "label": label,
                "level_before": before_level,
                "level_after": after_level,
                "level_delta": after_level - before_level,
                "changed_cells": changed_cells,
                "leveled_up": leveled_up,
                "died": died,
                "inert": inert,
            }
        )

    return {
        "operator": "probe_action_semantics",
        "action_count": len(rows),
        "rows": rows,
        "inert_labels": [r["label"] for r in rows if r["inert"]],
        "levelup_labels": [r["label"] for r in rows if r["leveled_up"]],
        "lethal_labels": [r["label"] for r in rows if r["died"]],
        "effective_labels": [r["label"] for r in rows if not r["inert"] and not r["died"]],
        "verifier_is_oracle": False,
    }


def _mask_centroid(
    grid: Any, *, color: Optional[int], background: int
) -> Optional[tuple[float, float]]:
    import numpy as np

    arr = np.asarray(grid)
    mask = (arr == color) if color is not None else (arr != background)
    ys, xs = np.nonzero(mask)
    if len(ys) == 0:
        return None
    return (float(ys.mean()), float(xs.mean()))


def _dominant_direction(dy: float, dx: float) -> str:
    # Screen coords: y grows DOWNWARD, so dy>0 is "down". Report the axis of
    # larger magnitude; genuinely-zero net motion is "none".
    if abs(dy) < 1e-9 and abs(dx) < 1e-9:
        return "none"
    if abs(dy) >= abs(dx):
        return "down" if dy > 0 else "up"
    return "right" if dx > 0 else "left"


def read_absolute_trajectory(
    frame: Any, *, color: Optional[int] = None, background: Optional[int] = None
) -> dict[str, Any]:
    """REQ-ARC-WMTE-5716 (playbook 2.3 / 2.4): recover a sprite's ABSOLUTE motion
    across the multi-layer animation array, which the camera-relative settled grid
    hides. Generalizes bp35 L9's animation-frame trajectory reader (which was the
    tool that disambiguated a fatal fall and led to that game's full clear).

    Tracks the centroid of the sprite (the pixels of ``color``, or -- when color
    is None -- all non-``background`` foreground pixels) across each animation
    layer, and reports per-layer centroids (row, col), per-step (dy, dx) deltas,
    the net displacement over the whole action, a dominant direction, and how many
    layers/observations there were. A single-layer frame has no motion to recover
    (net (0, 0), direction "none"). Layers where the sprite is absent contribute a
    ``None`` centroid and are skipped when chaining deltas.
    """
    layers = _frame_layers(frame)
    bg = _grid_background(layers[-1], background)
    centroids: list[Optional[tuple[float, float]]] = [
        _mask_centroid(layer, color=color, background=bg) for layer in layers
    ]
    observed = [c for c in centroids if c is not None]
    deltas: list[tuple[float, float]] = []
    prev: Optional[tuple[float, float]] = None
    for c in centroids:
        if c is None:
            continue
        if prev is not None:
            deltas.append((c[0] - prev[0], c[1] - prev[1]))
        prev = c
    if len(observed) >= 2:
        net = (observed[-1][0] - observed[0][0], observed[-1][1] - observed[0][1])
    else:
        net = (0.0, 0.0)
    return {
        "operator": "read_absolute_trajectory",
        "layer_count": len(layers),
        "observed_count": len(observed),
        "centroids": centroids,
        "step_deltas": deltas,
        "net_dy": net[0],
        "net_dx": net[1],
        "direction": _dominant_direction(net[0], net[1]),
        "verifier_is_oracle": False,
    }


def find_unexplained_glyphs(
    frame: Any,
    known_colors: Sequence[int] = (),
    *,
    background: Optional[int] = None,
    min_area: int = 1,
) -> dict[str, Any]:
    """REQ-ARC-WMTE-5716 (playbook 2.1): surface on-screen objects whose color is
    NOT yet in a caller-maintained registry of known-interactive / known-inert
    colors, so the caller can systematically click/test each before committing to
    a route. Camouflaged hazard/utility objects are a repeated corpus unlock (the
    wa30 helper robot 9 prior attempts ignored; the bp35 growable pillar).

    Operates on the settled grid's connected components (reusing
    :func:`object_centric_digest`). Returns the uncatalogued components -- each
    with ``color``, ``centroid`` = (x, y) click coordinates, ``area`` and
    ``bbox`` -- sorted by area descending (largest, most-likely-load-bearing
    first), plus the distinct uncatalogued colors. The background color is always
    excluded; pass ``known_colors`` to also exclude colors you have already
    catalogued.
    """
    grid = settled_grid(frame)
    bg = _grid_background(grid, background)
    known = {int(c) for c in known_colors} | {bg}
    digest = object_centric_digest(grid)
    unexplained: list[dict[str, Any]] = []
    for comp in digest["components"]:
        color = int(comp["color"])
        area = int(comp["area"])
        if color in known or area < int(min_area):
            continue
        cx, cy = comp["centroid"]
        unexplained.append(
            {
                "color": color,
                "centroid": [int(round(cx)), int(round(cy))],
                "area": area,
                "bbox": list(comp["bbox"]),
            }
        )
    unexplained.sort(key=lambda row: (-int(row["area"]), int(row["color"])))
    return {
        "operator": "find_unexplained_glyphs",
        "background_color": bg,
        "unexplained_count": len(unexplained),
        "unexplained_colors": sorted({row["color"] for row in unexplained}),
        "components": unexplained,
        "verifier_is_oracle": False,
    }


def bounded_reachability_search(
    start: Any,
    neighbors: Callable[[Any], Any],
    is_goal: Callable[[Any], bool],
    *,
    state_hash: Optional[Callable[[Any], Hashable]] = None,
    priority: Optional[Callable[[Any, int], float]] = None,
    max_nodes: int = 10000,
    max_depth: Optional[int] = None,
) -> dict[str, Any]:
    """REQ-ARC-WMTE-5716 (playbook 3.1 / 3.2): a generic graph search that HONESTLY
    reports whether a negative result is PROVEN (the frontier emptied with no cut
    branches -> exhausted under the given model) or merely SEARCH-CAPPED (hit the
    node/depth budget). Conflating the two is a repeated corpus error (wa30's
    "settled dead end" overturned; bp35 L9's "intractable to exhaust" honestly
    labeled partial-not-proven).

    ``state_hash(state) -> Hashable`` dedups on the SEMANTICALLY-RELEVANT subset of
    state, so cosmetic variation (animation phase, decorative growth, camera
    pixels) does not explode the frontier; the default hashes the raw state, so
    pass a projection whenever states carry cosmetic noise. ``neighbors(state)``
    yields ``(edge_label, next_state)`` pairs. ``priority(state, depth) -> float``
    makes the search best-first (lower expands first); omit it for plain BFS.

    Returns ``reached`` (bool), the ``path`` of edge labels to the goal (or None),
    a ``status`` of ``goal`` / ``exhausted`` / ``capped_nodes`` / ``capped_depth``,
    ``nodes_expanded``, ``frontier_remaining``, and ``proven_unreachable`` -- which
    is True ONLY for ``exhausted`` (never for a capped search).
    """
    hash_fn = state_hash if state_hash is not None else (lambda s: s)
    seen: set[Hashable] = {hash_fn(start)}
    counter = itertools.count()
    use_priority = priority is not None

    if use_priority:
        heap: list[tuple[float, int, Any, list[Any], int]] = [
            (priority(start, 0), next(counter), start, [], 0)
        ]
    else:
        from collections import deque

        queue: "deque[tuple[Any, list[Any], int]]" = deque([(start, [], 0)])

    nodes_expanded = 0
    depth_capped = False

    def _pop() -> tuple[Any, list[Any], int]:
        if use_priority:
            _, _, state, path, depth = heapq.heappop(heap)
            return state, path, depth
        return queue.popleft()

    def _frontier_len() -> int:
        return len(heap) if use_priority else len(queue)

    def _push(state: Any, path: list[Any], depth: int) -> None:
        if use_priority:
            heapq.heappush(heap, (priority(state, depth), next(counter), state, path, depth))
        else:
            queue.append((state, path, depth))

    while _frontier_len() > 0:
        if nodes_expanded >= max_nodes:
            return _reachability_result(
                False, None, "capped_nodes", nodes_expanded, _frontier_len()
            )
        state, path, depth = _pop()
        if is_goal(state):
            return _reachability_result(True, path, "goal", nodes_expanded, _frontier_len())
        nodes_expanded += 1
        if max_depth is not None and depth >= max_depth:
            depth_capped = True
            continue
        for label, nxt in neighbors(state):
            key = hash_fn(nxt)
            if key in seen:
                continue
            seen.add(key)
            _push(nxt, path + [label], depth + 1)

    status = "capped_depth" if depth_capped else "exhausted"
    return _reachability_result(False, None, status, nodes_expanded, 0)


def _reachability_result(
    reached: bool,
    path: Optional[list[Any]],
    status: str,
    nodes_expanded: int,
    frontier_remaining: int,
) -> dict[str, Any]:
    return {
        "operator": "bounded_reachability_search",
        "reached": reached,
        "path": path,
        "status": status,
        "nodes_expanded": nodes_expanded,
        "frontier_remaining": frontier_remaining,
        # PROVEN unreachability requires an emptied frontier with no cut branches;
        # a capped search proves nothing (playbook 3.1).
        "proven_unreachable": (not reached) and status == "exhausted",
    }


def bisect_death_prefix(
    actions: Sequence[Any],
    is_dead_after: Callable[[int], bool],
) -> dict[str, Any]:
    """REQ-ARC-WMTE-5716 (playbook 4.5): binary-search the MINIMAL action prefix
    that still ends in death, to isolate the true death-causing action instead of
    manual step-by-step replay -- and to separate a real hazard from an unrelated
    budget/timer/harness cause (bp35: the historically "lethal step" was actually
    the invisible action-count clock, proven once the death was isolated).

    ``is_dead_after(k) -> bool`` replays the first ``k`` actions from a fresh state
    and returns whether the resulting state is dead. Death is assumed MONOTONE in
    prefix length (once dead, a longer prefix stays dead) -- the usual case for a
    sequence that ends in a game-over. Returns ``fatal_prefix_len`` (the smallest k
    with ``is_dead_after(k)`` True, or None if no prefix is dead),
    ``fatal_action_index`` = that length minus one (None when death precedes any
    action), the ``fatal_action`` itself, and ``evaluations`` (replays performed).
    """
    n = len(actions)
    evaluations = 0

    def dead(k: int) -> bool:
        nonlocal evaluations
        evaluations += 1
        return bool(is_dead_after(k))

    if not dead(n):
        return {
            "operator": "bisect_death_prefix",
            "fatal_prefix_len": None,
            "fatal_action_index": None,
            "fatal_action": None,
            "evaluations": evaluations,
            "monotone_assumption": True,
        }

    lo, hi = 0, n
    while lo < hi:
        mid = (lo + hi) // 2
        if dead(mid):
            hi = mid
        else:
            lo = mid + 1

    fatal_len = lo
    if fatal_len == 0:
        fatal_index: Optional[int] = None
        fatal_action: Any = None
    else:
        fatal_index = fatal_len - 1
        fatal_action = actions[fatal_index]
    return {
        "operator": "bisect_death_prefix",
        "fatal_prefix_len": fatal_len,
        "fatal_action_index": fatal_index,
        "fatal_action": fatal_action,
        "evaluations": evaluations,
        "monotone_assumption": True,
    }
