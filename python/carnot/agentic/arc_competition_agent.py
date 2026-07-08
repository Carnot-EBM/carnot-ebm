"""CarnotAgent — an ARC Prize 2026 / ARC-AGI-3 competition agent (the Kaggle
submission shape). The competition runs an Agent subclass step-wise and OFFLINE:
each turn the harness calls `choose_action(frames, latest_frame) -> GameAction` and
`is_done(frames, latest_frame) -> bool`. No internet at eval time.

This module is FRAMEWORK-AGNOSTIC so it is testable offline without the
ARC-AGI-3-Agents package: the decision logic lives in `CarnotAgentPolicy`, and
`make_carnot_agent(Agent)` adapts it onto the real `Agent` base class at submission
time (a thin subclass). The validation harness `scripts/arc_competition_validate.py`
drives the policy through our offline sims (environment_files), mimicking the
competition loop, to confirm the agent scores BEFORE any submission.

Policy (v1 — recognize-and-replay): the harness gives the agent its `game_id` at
construction. For a game we have an OFFLINE-REPRODUCED solution (the 13-level
registry), the agent replays that banked action sequence (Mode-1; no search, no
internet — ideal for the offline eval IF the eval games == the public 25). For an
UNKNOWN game (hidden eval), it falls back to a step-wise systematic explorer
(navigate-by-RESET-replay, take untested salient actions) — the online form of
graph_explore_solve_v2. The replay path is the validated v1; the explore fallback is
the generalizing path for held-out games.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

import carnot.agentic.arc_strategy_router as arc_strategy_router
import carnot.agentic.arc_solve_learning as arc_solve_learning
import carnot.agentic.arc_discriminative_router as arc_discriminative_router
import carnot.agentic.arc_goal_energy_live as arc_goal_energy_live
from carnot.agentic.arc_amortized_exploration import coerce_amortized_first_contact_prior
from carnot.agentic.arc_color_blob_salience import ColorBlobSaliencePrior
from carnot.agentic.arc_dense_curiosity_progress import DenseCuriosityProgress
from carnot.agentic.arc_energy_fitness_qd import coerce_qd_generator
from carnot.agentic.arc_controllable_novelty import coerce_controllable_novelty_policy
from carnot.agentic.arc_go_explore import coerce_go_explore_archive
from carnot.agentic.arc_ige_cell_selector import coerce_ige_cell_selector
from carnot.agentic.arc_program_synthesis_filter import (
    coerce_program_synthesis_filter,
    induce_action_effect_proposal_filter,
)
from carnot.agentic.arc_frame_change_predictor import (
    ActionEffectExpansionPrior,
    GroundTruthValidatedFrameChangeScorer,
    load_live_action_effect_scorer,
)
from carnot.agentic.arc_goal_energy_live import GOAL_ENERGY_SOURCE
from carnot.agentic.arc_value_learner import coerce_object_centric_proposal_policy
from carnot.agentic.arc_value_net import load_live_spatial_value_head
from carnot.agentic.arc_world_model_dsl import ObjectDeltaModel
from carnot.agentic.arc_llm_reinduction import (
    MAX_REFINEMENT_ROUNDS,
    execute_bounded_llm_reinduction,
    plan_hierarchical_subgoals,
    propose_hierarchical_subgoals,
)
from carnot.agentic.arc_world_model_trust_energy import (
    HIDDEN_STATE_GAME_IDS,
    WorldModelCandidate,
    select_trusted_world_model,
)

REPO = Path(__file__).resolve().parents[3]

# the 11 reproduced games and their target (offline-reproduced) level
CLAIMED = {
    "r11l": 1,
    "lp85": 3,
    "ls20": 1,
    "wa30": 1,
    "cd82": 1,
    "sp80": 1,
    "su15": 1,
    "tu93": 1,
    "cn04": 1,
    "m0r0": 1,
    "sk48": 1,
}
MAX_ACTIONS = 200
# REQ-LEARN-4652: value_weight is raised off the 0.0 floor only after the component-labeling cost fix.
# The live route uses the cheap v2+frame-delta subset plus frame-hash caching, not full v3 per node.
SUBMITTED_VALUE_WEIGHT = 1e-12
SUBMITTED_VALUE_HEAD_FEATURE_SUBSET = "cross_game_features_v3:v2_plus_frame_delta"
DAGGER_VALUE_HEAD_RELATIVE_PATH = "models/arc_dagger_value_routing_v3.json"
# Exp 4605 wires the live scored path to attempt deeper levels. The verifier stays a tie-breaker
# (`value_weight=0`) so depth remains primary; deeper target levels only keep the loop alive after L1.
SUBMITTED_TARGET_LEVELS = 3
# Smart grace-period early-stop: stop this many moves after the LAST level-up if no new level appears
# (cuts the fruitless post-solve tail, WITHOUT capping the configured scored target). None = disabled.
SUBMITTED_EARLY_STOP_GRACE: Optional[int] = None
SUBMITTED_SEARCH_MODE = "depth_first_ride"
SUBMITTED_GRAPH_EXPLORE_BUDGET = 80
SUBMITTED_ROUTED_EXPLORE_BUDGET = 24
SUBMITTED_LAZY_VALUE_TOP_K = 4
SUBMITTED_FRONTIER_BATCH_SIZE: int | str = 1
SUBMITTED_NAVIGATION_COST_TIEBREAK = True
SUBMITTED_FRAME_CHANGE_PREDICTOR_ENABLED = True
SUBMITTED_FRAME_CHANGE_RANKING_MODE = "persistent_aem_plus_optional_cnn"
SUBMITTED_ACTION_EFFECT_EXPANSION_PRIOR_ENABLED = True
SUBMITTED_ACTION_EFFECT_EXPANSION_PRIOR_MODE = "persistent_aem_plus_optional_cnn_frontier_prior"
SUBMITTED_GOAL_ENERGY_ENABLED = True
SUBMITTED_GOAL_ENERGY_ALPHA = 0.9
SUBMITTED_GOAL_ENERGY_BETA = 0.1
SUBMITTED_GOAL_GUIDANCE_LAMBDA = 1.0
SUBMITTED_GOAL_ENERGY_CANDIDATE_GUIDANCE_ENABLED = True
SUBMITTED_GOAL_ENERGY_CANDIDATE_GUIDANCE_ALPHA = 0.0
SUBMITTED_GOAL_ENERGY_CANDIDATE_GUIDANCE_BETA = 1.0
SUBMITTED_QD_GENERATION_ENABLED = False
SUBMITTED_QD_GENERATION_MODE = "energy_fitness_map_elites_sequence_generator"
SUBMITTED_CONTROLLABLE_NOVELTY_PROPOSAL_ENABLED = False
SUBMITTED_CONTROLLABLE_NOVELTY_MODE = "episodic_knn_plus_rnd_action_effect_embedding"
SUBMITTED_OBJECT_CENTRIC_PROPOSAL_ENABLED = False
SUBMITTED_OBJECT_CENTRIC_PROPOSAL_MODE = "connected_component_slots_plus_relational_gaps"
SUBMITTED_COLOR_BLOB_SALIENCE_ENABLED = True
SUBMITTED_COLOR_BLOB_SALIENCE_MODE = "single_color_connected_component_tiers"
SUBMITTED_PROGRAM_SYNTHESIS_PROPOSAL_FILTER_ENABLED = False
SUBMITTED_PROGRAM_SYNTHESIS_PROPOSAL_FILTER_TRUST_THRESHOLD = 0.75
SUBMITTED_AMORTIZED_FIRST_CONTACT_PRIOR_ENABLED = False
SUBMITTED_AMORTIZED_FIRST_CONTACT_PRIOR_MODE = (
    "frequency_prior_from_cross_game_first_contact_traces"
)
SUBMITTED_GO_EXPLORE_ARCHIVE_ENABLED = False
SUBMITTED_GO_EXPLORE_ARCHIVE_MODE = "return_then_explore_replayable_prefix_archive"
# IGE-style LLM-guided cell selection on top of the Go-Explore archive (Intelligent Go-Explore,
# arXiv:2405.15143). OFF by default: it is an OPEN-question lever (the plain archive nulled on first-win;
# this swaps the cell-choice heuristic for an LLM promisingness judge). Enable only with the archive on.
SUBMITTED_IGE_CELL_SELECTION_ENABLED = False
SUBMITTED_IGE_CELL_SELECTION_MODE = "llm_promisingness_go_explore"
# REQ-ARC-WMTE-4933: MATM-style similarity retrieval is an opt-in efficiency lever. The submitted
# baseline remains exact frame-hash navigation until the Experiment 4933 gate proves a strict gain.
SUBMITTED_MATM_SIMILARITY_RETRIEVAL_ENABLED = False
SUBMITTED_MATM_SIMILARITY_RETRIEVAL_MODE = "within_game_lsh_cross_game_features_v2"
MATM_SIMILARITY_BUCKET_WIDTH = 0.25
MATM_SIMILARITY_MAX_CANDIDATES = 8
_DEFAULT_VALUE_HEAD = object()
_DEFAULT_CANDIDATE_ROUTER = object()
_DEFAULT_FRAME_CHANGE_SCORER = object()
_DEFAULT_GOAL_BIAS = object()


def _object_identity_perception_hook() -> type:
    """REQ-ARC-WMTE-4841: keep the prototype perception layer live-path importable."""

    from carnot.agentic.arc_object_identity_perception import TrackerConfig

    return TrackerConfig


def load_solutions() -> dict[str, list[dict]]:
    """game-short -> [{"action": int, "data": {x,y}|None}] for every banked solution,
    via the metaharness's loader (single source of truth for the trajectories)."""
    spec = importlib.util.spec_from_file_location(
        "mh", str(REPO / "scripts" / "arc3_replay_scorecard_metaharness.py")
    )
    mh = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mh)  # type: ignore
    sols: dict[str, list[dict]] = {}
    for short in CLAIMED:
        src = mh.RESOLVED_ARTIFACTS.get(short, mh.GAME_ARTIFACTS.get(short))
        if not src:
            continue
        steps = []
        for a in mh.load_actions(src):
            aid, data = mh.normalize(a)
            if aid is not None:
                steps.append({"action": int(aid), "data": data})
        sols[short] = steps
    return sols


def _level_of(frame: Any) -> int:
    if frame is None:
        return 0
    from carnot.agentic.arc_agi3_live_adapter import _levels_completed

    try:
        return _levels_completed(frame)
    except Exception:
        return 0


def _action_key(action_id: int, data: Any) -> tuple:
    if int(action_id) == 6 and isinstance(data, dict) and "x" in data and "y" in data:
        return (6, int(data["x"]), int(data["y"]))
    return (int(action_id),)


def _frame_only_mechanic_hint(frame: Any) -> Optional[str]:
    """Cheap frame-only mechanic hint used when the live route has no registry class.

    The full probe-based detector lives at the I/O edge (`scripts/arc3_frame_induction.py`).
    The submitted policy cannot touch `env._game`, so this hint only inspects the rendered
    grid and returns a positive class when a dense lower-board editor palette is visible.
    Unknown frames return None and keep the registry/default route.
    """
    if frame is None:
        return None
    try:
        import numpy as np
        from carnot.agentic.arc_agi3_world_model import grid_of

        grid = grid_of(frame)
        if getattr(grid, "shape", None) is None or grid.ndim != 2:
            return None
        h, w = grid.shape
        lower = grid[int(h * 0.55) :, :]
        if lower.size == 0:
            return None
        vals, counts = np.unique(lower, return_counts=True)
        bg = int(vals[counts.argmax()])
        active = lower != bg
        density = float(active.mean())
        columns = int(np.count_nonzero(active.any(axis=0)))
        rows = int(np.count_nonzero(active.any(axis=1)))
        if density >= 0.08 and columns >= max(12, w // 5) and rows >= 5:
            return "program_editor"
    except Exception:
        return None
    return None


def _route_explore_budget(route: dict[str, Any]) -> int:
    if route.get("uses_goal_distance_heuristic") is False:
        return SUBMITTED_ROUTED_EXPLORE_BUDGET
    return SUBMITTED_GRAPH_EXPLORE_BUDGET


def _recommend_live_approach(game_id: str, *, mechanic: Optional[str] = None) -> dict[str, Any]:
    """Return the solve-learning recommendation, falling back to the lightweight strategy router."""

    try:
        rec = arc_solve_learning.recommend_approach(game_id, mechanic=mechanic)
        if isinstance(rec, dict) and isinstance(rec.get("strategy"), dict):
            return rec
    except Exception as exc:
        return {
            "error": f"recommend_approach_failed:{type(exc).__name__}",
            "strategy": arc_strategy_router.route_for_game(game_id, mechanic=mechanic),
        }
    return {"strategy": arc_strategy_router.route_for_game(game_id, mechanic=mechanic)}


def _load_submitted_candidate_router() -> Any | None:
    """Load the Exp4545 v3 discriminative router as a safe candidate-order tie-breaker."""

    try:
        return arc_discriminative_router.load_cross_game_discriminative_router(root=REPO)
    except Exception:
        return None


def _load_submitted_frame_change_scorer() -> Any | None:
    """REQ-ARC-FCP-4629/5373: load the validated live action-effect scorer."""

    if not SUBMITTED_FRAME_CHANGE_PREDICTOR_ENABLED:
        return None
    try:
        scorer = load_live_action_effect_scorer(root=REPO)
        if scorer is None:
            return None
        return GroundTruthValidatedFrameChangeScorer(scorer)
    except Exception:
        return None


def _load_submitted_goal_energy_bias() -> Any | None:
    """REQ-ARC-WMTE-4640: load Exp4020's graded visible-state goal energy."""

    if not SUBMITTED_GOAL_ENERGY_ENABLED:
        return None
    return arc_goal_energy_live.load_exp4020_goal_energy(root=REPO)


class StepwiseExplorer:
    """Generic, game-AGNOSTIC step-wise solver — the competition asset (eval games are
    UNSEEN, so replay is useless; this is the only thing that scores). It is
    `graph_explore_solve_v2` turned inside-out into an EVENT LOOP the harness drives one
    action per turn: maintain a state-transition graph, expand the current state's
    untested SALIENT actions DEPTH-first (no navigation cost), and when the current
    state is exhausted or dead-ends, navigate to the shallowest frontier state with
    untested actions by RESET + replaying its path (deepcopy/jump is impossible live;
    RESET-replay is the only navigation). Action-efficient (scoring rewards fewer
    actions: min(human/agent,1)^2). Stops at the first level-up (+1 incremental unit) or
    when fully explored."""

    def __init__(
        self,
        target_levels: int = 1,
        max_depth: int = 45,
        hud_mask=None,
        value_head=None,
        value_weight: float = 0.0,
        search_mode: str = "depth_first_ride",
        online_discriminative: bool = True,
        discriminative_featurizer=None,
        discriminative_min_positives: int = 3,
        discriminative_min_negatives: int = 3,
        discriminative_fit_iters: int = 120,
        discriminative_refit_every: int = 4,
        discriminative_prune_threshold: float = 0.12,
        frame_change_scorer: Any | None = None,
        frame_change_prune_threshold: float | None = None,
        action_effect_expansion_prior: Any | bool | None = None,
        action_prior: Any | None = None,
        action_prior_prune_quantile: float | None = None,
        adaptive_budget_threshold: float | None = None,
        adaptive_budget_value_head: Any | None = None,
        adaptive_budget_noop_threshold: float = 0.5,
        lazy_value_top_k: int = SUBMITTED_LAZY_VALUE_TOP_K,
        early_stop_grace: Optional[int] = None,
        frontier_batch_size: int | str | None = SUBMITTED_FRONTIER_BATCH_SIZE,
        navigation_cost_tiebreak: bool = SUBMITTED_NAVIGATION_COST_TIEBREAK,
        candidate_router: Any | None = None,
        dense_curiosity: bool | DenseCuriosityProgress = False,
        dense_curiosity_weight: float = 0.15,
        dense_curiosity_discount: float = 0.5,
        goal_bias: Any | None = None,
        goal_bias_label: str = "",
        goal_bias_lower_is_better: bool = True,
        goal_candidate_guidance: Any | None = None,
        qd_generator: Any | bool | None = None,
        controllable_novelty: Any | bool | None = SUBMITTED_CONTROLLABLE_NOVELTY_PROPOSAL_ENABLED,
        object_centric_proposal: Any | bool | None = SUBMITTED_OBJECT_CENTRIC_PROPOSAL_ENABLED,
        program_synthesis_filter: Any | None = None,
        amortized_first_contact_prior: Any | bool | None = (
            SUBMITTED_AMORTIZED_FIRST_CONTACT_PRIOR_ENABLED
        ),
        go_explore_archive: Any | bool | None = SUBMITTED_GO_EXPLORE_ARCHIVE_ENABLED,
        similarity_retrieval: bool | None = None,
        similarity_bucket_width: float = MATM_SIMILARITY_BUCKET_WIDTH,
        similarity_max_candidates: int = MATM_SIMILARITY_MAX_CANDIDATES,
    ) -> None:
        self.hud_mask = hud_mask  # E1: mask step-counter cells out of node identity
        # BRIDGE: a frame-only cross-game value head (frame -> predicted steps-to-next-level-up, LOWER =
        # closer). Trained offline on ALL banked solves (the offline->live distillation). The frontier is
        # ordered A*-style: priority = depth + value_weight*value. value_weight=0 (default) -> pure BFS
        # (value only breaks ties; can't regress). value_weight>0 -> the value head NUDGES the search
        # toward predicted-closer states (the A* routing that unlocked cn04 in graph_explore_solve_v2).
        self.value_head = value_head
        self.value_weight = float(value_weight)
        self.lazy_value_top_k = max(1, int(lazy_value_top_k))
        self._value_cache: dict[str, float] = {}
        self._value_head_evals = 0
        self._value_cache_hits = 0
        self.frame_change_scorer = frame_change_scorer
        self.frame_change_prune_threshold = frame_change_prune_threshold
        if hasattr(action_effect_expansion_prior, "frontier_priority"):
            self.action_effect_expansion_prior = action_effect_expansion_prior
        elif action_effect_expansion_prior and frame_change_scorer is not None:
            self.action_effect_expansion_prior = ActionEffectExpansionPrior(frame_change_scorer)
        else:
            self.action_effect_expansion_prior = None
        self.action_prior = action_prior
        self.action_prior_prune_quantile = action_prior_prune_quantile
        self.adaptive_budget_threshold = adaptive_budget_threshold
        self.adaptive_budget_value_head = adaptive_budget_value_head
        self.adaptive_budget_noop_threshold = float(adaptive_budget_noop_threshold)
        self._adaptive_budget_history: list[dict[str, Any]] = []
        self._adaptive_budget_commit_count = 0
        self._adaptive_budget_expanded_count = 0
        self._adaptive_budget_candidates_skipped = 0
        self.online_discriminative = bool(online_discriminative)
        self.discriminative_featurizer = discriminative_featurizer
        self.discriminative_min_positives = max(1, int(discriminative_min_positives))
        self.discriminative_min_negatives = max(1, int(discriminative_min_negatives))
        self.discriminative_fit_iters = max(1, int(discriminative_fit_iters))
        self.discriminative_refit_every = max(1, int(discriminative_refit_every))
        self.discriminative_prune_threshold = float(discriminative_prune_threshold)
        self.online_discriminator = None
        self._disc_X: list[list[float]] = []
        self._disc_y: list[float] = []
        self._disc_seen: set[tuple[int, str]] = set()
        self._disc_rows: list[dict[str, Any]] = []
        self._disc_samples_since_fit = 0
        self._disc_negative_sources: dict[str, int] = {}
        self._disc_fit_count = 0
        self._disc_frontier_pruned = 0
        # SEARCH MODE: "depth_first_ride" (default, proven) rides the current branch depth-first (action-
        # efficient; load-bearing for the deep wins lp85/sp80). "best_first" ALWAYS expands the globally-
        # best A*-value frontier (depth + value_weight*value) -- this is the graph_explore_solve_v2 search
        # form where the cross-game value head's routing actually helped (it unlocked cn04). best_first
        # only beats the ride when the value head is good enough to route the deep wins itself; measure it.
        self.search_mode = search_mode
        self.graph: dict[
            str, dict
        ] = {}  # hash -> {"path": [...], "untested": [...], "value": float}
        self.root: Optional[str] = None
        self.cur: Optional[str] = None
        self.start_level: Optional[int] = None
        self.best_level = 0
        self.target_levels = target_levels  # stop after this many levels beyond start
        self.max_depth = max_depth  # cap DFS branch length -> forces backtrack
        self.pending: list[dict] = []  # queued nav/probe actions
        self.awaiting: Optional[dict] = None  # last probe, to attribute its result
        self.explored_out = False
        self.frontier_batch_size = self._normalize_frontier_batch_size(frontier_batch_size)
        self.navigation_cost_tiebreak = bool(navigation_cost_tiebreak)
        self.candidate_router = candidate_router
        if isinstance(dense_curiosity, DenseCuriosityProgress):
            self.dense_curiosity: DenseCuriosityProgress | None = dense_curiosity
        elif dense_curiosity:
            self.dense_curiosity = DenseCuriosityProgress(
                "?",
                bonus_weight=dense_curiosity_weight,
                backup_discount=dense_curiosity_discount,
            )
        else:
            self.dense_curiosity = None
        # HYBRID exploration diversity (flag-gated, default OFF -> byte-identical/parity-preserving). The
        # depth_first_ride over-commits to the top-salient branch and MISSES easy "structure-missed" wins
        # (r11l/sp80: 0/2000 structured but trivially reachable randomly). With CARNOT_ARC_EXPLORE_DIVERSITY=1,
        # once the search has STALLED (no new level for _stall_threshold moves), pop a RANDOM untested action
        # among the top-K instead of the most-salient pop(0) -- recovering the structure-missed tail without
        # costing the efficient wins. Measured: hybrid 3/11 vs structured 1/11, lp85 kept efficient (eff 2.0069).
        import os as _os
        import random as _random

        self._hybrid_diversity = _os.environ.get("CARNOT_ARC_EXPLORE_DIVERSITY", "0") != "0"
        self._stall_threshold = int(_os.environ.get("CARNOT_ARC_EXPLORE_STALL", "150"))
        self._div_topk = int(_os.environ.get("CARNOT_ARC_EXPLORE_DIV_TOPK", "8"))
        self._steps_since_progress = 0
        self._nm_best_level = 0
        self._div_rng = _random.Random(20260621)
        # Smart grace-period early-stop (does NOT cap levels). After reaching >=1 level, keep searching
        # for the next; stop only if no NEW level within `early_stop_grace` moves of the last level-up.
        # Consecutive level-ups reset the window, so multi-level games are NOT capped -- only the fruitless
        # tail after the last findable level is cut (that tail destroys the (human/actions)^2 score: e.g.
        # lp85 reaches L1 at action 20 but ran to 7792 hunting unreachable deeper levels). None = disabled.
        self.early_stop_grace = early_stop_grace
        self._early_stop_level_mark = 0
        self._early_stop_frame_mark = 0
        self.early_stopped = False
        self.adj: dict[str, list] = {}  # known forward edges: hash -> [(action_dict, next_hash)]
        self._nav_attempts = 0
        self._nav_exact_hits = 0
        self._nav_partial_hits = 0
        self._nav_reset_fallbacks = 0
        self._nav_edges_recorded = 0
        self._nav_forward_steps = 0
        self._nav_reset_replay_steps = 0
        if similarity_retrieval is None:
            similarity_retrieval = (
                _os.environ.get("CARNOT_ARC_MATM_SIMILARITY_RETRIEVAL", "0") == "1"
            )
        self.similarity_retrieval_enabled = bool(similarity_retrieval)
        self.similarity_bucket_width = max(1e-9, float(similarity_bucket_width))
        self.similarity_max_candidates = max(1, int(similarity_max_candidates))
        self._similarity_state_buckets: dict[tuple[int, ...], list[str]] = {}
        self._similarity_descriptor_by_hash: dict[str, tuple[int, ...]] = {}
        self._last_shortest_path_kind: str | None = None
        self._nav_similarity_hits = 0
        self._nav_similarity_candidates_considered = 0
        self._nav_similarity_router_accepts = 0
        self._nav_similarity_router_rejects = 0
        self._nav_similarity_value_checks = 0
        self._nav_similarity_goal_checks = 0
        self._nav_similarity_world_model_verifier_checks = 0
        self.goal_bias = goal_bias
        self.goal_bias_label = str(goal_bias_label or "")
        self.goal_bias_lower_is_better = bool(goal_bias_lower_is_better and goal_bias is not None)
        self._goal_bias_scored = 0
        self._goal_bias_errors = 0
        self.goal_candidate_guidance = goal_candidate_guidance
        self.qd_generator = coerce_qd_generator(
            qd_generator,
            action_effect_scorer=self.frame_change_scorer,
            goal_energy=self.goal_bias,
        )
        self.controllable_novelty_policy = coerce_controllable_novelty_policy(
            controllable_novelty,
            action_effect_scorer=self.frame_change_scorer,
        )
        self.object_centric_proposal_policy = coerce_object_centric_proposal_policy(
            object_centric_proposal
        )
        self.program_synthesis_filter = coerce_program_synthesis_filter(program_synthesis_filter)
        self.amortized_first_contact_prior = coerce_amortized_first_contact_prior(
            amortized_first_contact_prior
        )
        self.go_explore_archive = coerce_go_explore_archive(go_explore_archive)
        # IGE cell selection (2026-06-28): when the flag is on and a Go-Explore archive exists without an
        # explicit selector, attach the LLM-promisingness cell selector (Intelligent Go-Explore). The
        # selector only RANKS already-archived cells (verifier_is_oracle=False) and falls back to the
        # archive heuristic if the GPU server / parse fails, so flipping it on cannot fabricate a solve.
        if (
            SUBMITTED_IGE_CELL_SELECTION_ENABLED
            and self.go_explore_archive is not None
            and getattr(self.go_explore_archive, "selector", None) is None
        ):
            self.go_explore_archive.selector = coerce_ige_cell_selector(True)
        self._qd_sequences_injected = 0
        self._qd_actions_injected = 0
        self._qd_generation_errors = 0
        self._go_explore_prefixes_injected = 0
        self._go_explore_actions_injected = 0

    @staticmethod
    def _normalize_frontier_batch_size(value: int | str | None) -> int | None:
        """REQ-ARC-FCP-4523: normalize k for the opt-in frontier batch sweep."""

        if value is None:
            return 1
        if isinstance(value, str) and value.lower() == "all":
            return None
        return max(1, int(value))

    def _disc_features(self, frame, previous_frame: Any | None = None) -> Optional[list[float]]:
        if not self.online_discriminative:
            return None
        try:
            if self.discriminative_featurizer is None:
                from carnot.agentic.arc_value_learner import cross_game_features_v2

                self.discriminative_featurizer = cross_game_features_v2
            try:
                values = self.discriminative_featurizer(
                    frame,
                    previous_frame=previous_frame,
                )
            except TypeError:
                values = self.discriminative_featurizer(frame)
            return [float(v) for v in values]
        except Exception:
            return None

    def _record_discriminative_sample(
        self,
        frame,
        *,
        previous_frame: Any | None = None,
        label: int,
        source: str,
        node_hash: Optional[str] = None,
        path: Sequence[Mapping[str, Any]] | None = None,
    ) -> Optional[list[float]]:
        features = self._disc_features(frame, previous_frame=previous_frame)
        if features is None:
            return None
        self._record_discriminative_features(
            features,
            label=label,
            source=source,
            node_hash=node_hash or self._hash(frame),
            path=path,
        )
        return features

    def _record_discriminative_features(
        self,
        features: Sequence[float] | None,
        *,
        label: int,
        source: str,
        node_hash: str,
        path: Sequence[Mapping[str, Any]] | None = None,
    ) -> None:
        if features is None:
            return
        key = (int(label), node_hash)
        if key in self._disc_seen:
            return
        self._disc_seen.add(key)
        self._disc_X.append([float(v) for v in features])
        self._disc_y.append(float(label))
        self._disc_samples_since_fit += 1
        if int(label) == 0:
            self._disc_negative_sources[source] = self._disc_negative_sources.get(source, 0) + 1
        self._disc_rows.append(
            {
                "features": [float(v) for v in features],
                "label": float(label),
                "source": str(source),
                "node_hash": str(node_hash),
                "path": [
                    {"action": int(step["action"]), "data": step.get("data")}
                    for step in (path or [])
                    if isinstance(step, Mapping) and step.get("action") is not None
                ],
            }
        )
        self._maybe_fit_discriminator()

    def _maybe_fit_discriminator(self) -> None:
        if not self.online_discriminative or not self._disc_y:
            return
        positives = int(sum(1 for y in self._disc_y if y >= 0.5))
        negatives = len(self._disc_y) - positives
        if positives < self.discriminative_min_positives:
            return
        if negatives < self.discriminative_min_negatives:
            return
        if (
            self.online_discriminator is not None
            and self._disc_samples_since_fit < self.discriminative_refit_every
        ):
            return
        try:
            from carnot.agentic.arc_value_learner import DiscriminativeVerifier

            verifier = DiscriminativeVerifier(lambda frame: self._disc_features(frame) or [])
            verifier.fit(
                self._disc_X,
                self._disc_y,
                iters=self.discriminative_fit_iters,
                lr=0.3,
                l2=1e-3,
            )
            self.online_discriminator = verifier
            self._disc_fit_count += 1
            self._disc_samples_since_fit = 0
        except Exception:
            return

    def _node_on_path_proba(self, node: dict) -> float:
        if self.online_discriminator is None:
            return 1.0
        features = node.get("discriminative_features")
        if features is None:
            return 1.0
        try:
            return float(self.online_discriminator.proba_features(features))
        except Exception:
            return 1.0

    def online_discriminator_diagnostics(self) -> dict[str, Any]:
        positives = int(sum(1 for y in self._disc_y if y >= 0.5))
        negatives = len(self._disc_y) - positives
        return {
            "enabled": bool(self.online_discriminative),
            "trained": self.online_discriminator is not None,
            "positive_samples": positives,
            "negative_samples": negatives,
            "negative_sources": dict(self._disc_negative_sources),
            "fit_count": int(self._disc_fit_count),
            "frontier_pruned": int(self._disc_frontier_pruned),
            "prune_threshold": float(self.discriminative_prune_threshold),
        }

    def search_distribution_samples(self) -> list[dict[str, Any]]:
        """REQ-LEARN-4665: expose live frontier samples for DAgger-lite aggregation."""

        return [dict(row) for row in self._disc_rows]

    def adaptive_budget_diagnostics(self) -> dict[str, Any]:
        return {
            "enabled": self.adaptive_budget_threshold is not None,
            "threshold": self.adaptive_budget_threshold,
            "commit_count": int(self._adaptive_budget_commit_count),
            "expanded_count": int(self._adaptive_budget_expanded_count),
            "candidates_skipped": int(self._adaptive_budget_candidates_skipped),
            "history": list(self._adaptive_budget_history[-64:]),
        }

    def lazy_value_diagnostics(self) -> dict[str, Any]:
        return {
            "enabled": self.value_head is not None and self.value_weight != 0.0,
            "lazy_top_k": int(self.lazy_value_top_k),
            "cache_by_frame_hash": True,
            "value_head_evals": int(self._value_head_evals),
            "cache_hits": int(self._value_cache_hits),
            "cached_frame_hashes": int(len(self._value_cache)),
        }

    def navigation_diagnostics(self) -> dict[str, Any]:
        """SCENARIO-ARC-FCP-4516: expose whether frontier navigation avoids RESET replay."""

        hits = int(self._nav_exact_hits + self._nav_partial_hits)
        hits += int(self._nav_similarity_hits)
        attempts = int(self._nav_attempts)
        return {
            "navigation_attempts": attempts,
            "exact_shortest_path_hits": int(self._nav_exact_hits),
            "partial_forward_walk_hits": int(self._nav_partial_hits),
            "similarity_forward_walk_hits": int(self._nav_similarity_hits),
            "forward_walk_hits": hits,
            "reset_replay_fallbacks": int(self._nav_reset_fallbacks),
            "forward_edges_recorded": int(self._nav_edges_recorded),
            "forward_navigation_steps": int(self._nav_forward_steps),
            "reset_replay_steps": int(self._nav_reset_replay_steps),
            "forward_walk_hit_rate": float(hits / attempts) if attempts else 0.0,
            "similarity_retrieval_enabled": bool(self.similarity_retrieval_enabled),
            "similarity_buckets": int(len(self._similarity_state_buckets)),
            "similarity_indexed_states": int(len(self._similarity_descriptor_by_hash)),
            "similarity_candidates_considered": int(self._nav_similarity_candidates_considered),
            "similarity_router_accepts": int(self._nav_similarity_router_accepts),
            "similarity_router_rejects": int(self._nav_similarity_router_rejects),
            "similarity_value_checks": int(self._nav_similarity_value_checks),
            "similarity_goal_checks": int(self._nav_similarity_goal_checks),
            "similarity_world_model_verifier_checks": int(
                self._nav_similarity_world_model_verifier_checks
            ),
        }

    def set_goal_bias(self, goal_bias, *, label: str = "", lower_is_better: bool = False) -> None:
        """REQ-ARC-WMTE-4533/4534: install a depth-preserving goal or energy bias."""

        self.goal_bias = goal_bias
        self.goal_bias_label = str(label or "")
        self.goal_bias_lower_is_better = bool(lower_is_better and goal_bias is not None)
        if self.goal_candidate_guidance is not None and hasattr(
            self.goal_candidate_guidance,
            "set_goal_energy",
        ):
            try:
                self.goal_candidate_guidance.set_goal_energy(goal_bias)
            except Exception:
                pass

    def goal_bias_diagnostics(self) -> dict[str, Any]:
        return {
            "enabled": self.goal_bias is not None,
            "label": self.goal_bias_label,
            "lower_is_better": bool(self.goal_bias_lower_is_better),
            "nodes_scored": int(self._goal_bias_scored),
            "errors": int(self._goal_bias_errors),
        }

    def goal_candidate_guidance_diagnostics(self) -> dict[str, Any]:
        """REQ-ARC-WMTE-4737: expose candidate-state goal-energy guidance diagnostics."""

        if self.goal_candidate_guidance is None:
            return {"enabled": False}
        if hasattr(self.goal_candidate_guidance, "diagnostics"):
            return dict(self.goal_candidate_guidance.diagnostics())
        return {"enabled": True}

    def action_effect_expansion_prior_diagnostics(self) -> dict[str, Any]:
        if self.action_effect_expansion_prior is None:
            return {"enabled": False}
        if hasattr(self.action_effect_expansion_prior, "diagnostics"):
            return dict(self.action_effect_expansion_prior.diagnostics())
        return {"enabled": True}

    def curiosity_diagnostics(self) -> dict[str, Any]:
        if self.dense_curiosity is None:
            return {"enabled": False}
        return self.dense_curiosity.diagnostics()

    def qd_generation_diagnostics(self) -> dict[str, Any]:
        diagnostics = {
            "enabled": self.qd_generator is not None,
            "sequences_injected": int(self._qd_sequences_injected),
            "actions_injected": int(self._qd_actions_injected),
            "generation_errors": int(self._qd_generation_errors),
            "verifier_is_oracle": False,
        }
        if self.qd_generator is not None and hasattr(self.qd_generator, "diagnostics"):
            diagnostics["generator"] = self.qd_generator.diagnostics()
        return diagnostics

    def controllable_novelty_diagnostics(self) -> dict[str, Any]:
        """REQ-ARC-WMTE-4688: expose proposal-novelty gate and memory diagnostics."""

        if self.controllable_novelty_policy is None:
            return {"enabled": False}
        return self.controllable_novelty_policy.diagnostics()

    def object_centric_proposal_diagnostics(self) -> dict[str, Any]:
        """REQ-ARC-WMTE-4700: expose object-slot proposal diagnostics."""

        if self.object_centric_proposal_policy is None:
            return {"enabled": False}
        return self.object_centric_proposal_policy.diagnostics()

    def amortized_prior_diagnostics(self) -> dict[str, Any]:
        """REQ-ARC-WMTE-4701: expose first-contact prior diagnostics."""

        if self.amortized_first_contact_prior is None:
            return {"enabled": False}
        return self.amortized_first_contact_prior.diagnostics()

    def go_explore_archive_diagnostics(self) -> dict[str, Any]:
        """REQ-ARC-WMTE-4701: expose return-then-explore prefix archive diagnostics."""

        if self.go_explore_archive is None:
            return {"enabled": False}
        out = self.go_explore_archive.diagnostics()
        out["prefixes_injected"] = int(self._go_explore_prefixes_injected)
        out["actions_injected"] = int(self._go_explore_actions_injected)
        return out

    def set_program_synthesis_filter(self, proposal_filter: Any | None) -> None:
        """REQ-ARC-WMTE-4689: install held-out-validated action-effect pruning."""

        self.program_synthesis_filter = coerce_program_synthesis_filter(proposal_filter)

    def program_synthesis_filter_diagnostics(self) -> dict[str, Any]:
        """REQ-ARC-WMTE-4689: expose held-out counts and pruning diagnostics."""

        if self.program_synthesis_filter is None:
            return {"enabled": False}
        return self.program_synthesis_filter.diagnostics()

    def _curiosity_score(self, node_hash: str) -> float:
        if self.dense_curiosity is None:
            return 0.0
        return self.dense_curiosity.score_state(node_hash)

    def _goal_bias_score(self, node: Mapping[str, Any]) -> float:
        if self.goal_bias is None:
            return 0.0
        frame = node.get("frame")
        if frame is None:
            return 0.0
        try:
            score = float(self.goal_bias(frame))
            self._goal_bias_scored += 1
            return score
        except Exception:
            self._goal_bias_errors += 1
            return 0.0

    def _goal_bias_key(self, score: float) -> float:
        if self.goal_bias is None:
            return 0.0
        if self.goal_bias_lower_is_better:
            return float(score)
        return -float(score)

    def _action_effect_frontier_key(self, node: Mapping[str, Any]) -> float:
        if self.action_effect_expansion_prior is None:
            return 0.0
        frame = node.get("frame")
        if frame is None:
            return 0.0
        try:
            return float(
                self.action_effect_expansion_prior.frontier_priority(
                    frame,
                    node.get("untested") or [],
                )
            )
        except Exception:
            return 0.0

    def _hash(self, frame) -> str:
        from carnot.agentic.arc_agi3_world_model import grid_of, frame_hash

        g = grid_of(frame)
        if self.hud_mask is not None and getattr(self.hud_mask, "shape", None) == g.shape:
            g = g.copy()
            g[self.hud_mask] = 0  # collapse counter/timer cells so equal game states dedup
        return frame_hash(g)

    def _grid_for_hash(self, node_hash: Optional[str]):
        if node_hash is None:
            return None
        node = self.graph.get(node_hash)
        frame = node.get("frame") if node else None
        if frame is None:
            return None
        try:
            from carnot.agentic.arc_agi3_world_model import grid_of

            return grid_of(frame)
        except Exception:
            return None

    def _candidates(
        self,
        frame,
        path: Sequence[dict] | None = None,
        previous_frame: Any | None = None,
    ) -> list[dict]:
        from carnot.agentic.arc_graph_explore import rich_action_candidates

        action_prior = self.action_prior
        if action_prior is not None and hasattr(action_prior, "for_path"):
            action_prior = action_prior.for_path(path or [])
        candidates = rich_action_candidates(
            frame,
            frame_change_scorer=self.frame_change_scorer,
            frame_change_prune_threshold=self.frame_change_prune_threshold,
            action_prior=action_prior,
            action_prior_prune_quantile=self.action_prior_prune_quantile,
            candidate_router=self.candidate_router,
            previous_frame=previous_frame,
        )
        if self.adaptive_budget_threshold is not None and candidates:
            from carnot.agentic.arc_adaptive_budget import apply_adaptive_budget

            try:
                frame_hash = self._hash(frame)
                frame_is_novel = frame_hash not in self.graph
            except Exception:
                frame_is_novel = True
            gated, decision = apply_adaptive_budget(
                frame,
                candidates,
                threshold=self.adaptive_budget_threshold,
                value_head=self.adaptive_budget_value_head or self.value_head,
                frame_change_scorer=self.frame_change_scorer,
                frame_is_novel=frame_is_novel,
                change_threshold=self.adaptive_budget_noop_threshold,
            )
            if decision.committed_single_candidate:
                self._adaptive_budget_commit_count += 1
                self._adaptive_budget_candidates_skipped += max(
                    0,
                    int(decision.normal_width) - int(decision.budget),
                )
            else:
                self._adaptive_budget_expanded_count += 1
            self._adaptive_budget_history.append(decision.as_dict())
            candidates = gated
        rows = [{"action": int(c.action_id), "data": c.data} for c in candidates]
        # PROPOSE hook (REQ-ARC-OAE-4710): inject click-heatmap proposals from an online scorer
        # that has propose_enabled=True.  `getattr(fcs, "propose_enabled", False)` guarantees
        # the LiveActionEffectScorer (frozen shipped scorer) is a no-op here -- it has no
        # propose_enabled attribute so the default False short-circuits the whole block.
        fcs = self.frame_change_scorer
        if (
            fcs is not None
            and getattr(fcs, "propose_enabled", False)
            and hasattr(fcs, "propose_coords")
        ):
            try:
                existing = {
                    (
                        int(r["action"]),
                        (r.get("data") or {}).get("x"),
                        (r.get("data") or {}).get("y"),
                    )
                    for r in rows
                }
                for px, py in fcs.propose_coords(frame):
                    key = (6, px, py)
                    if key not in existing:
                        rows.append({"action": 6, "data": {"x": int(px), "y": int(py)}})
                        existing.add(key)
            except Exception:
                pass
        rows = self._apply_amortized_prior_order(frame, rows, path=path)
        if self.object_centric_proposal_policy is not None:
            rows = self._apply_object_centric_proposal_order(
                frame,
                rows,
                previous_frame=previous_frame,
            )
        if self.program_synthesis_filter is not None:
            rows = self.program_synthesis_filter.rank_candidates(frame, rows)
        if self.goal_candidate_guidance is not None and rows:
            try:
                rows = self.goal_candidate_guidance.rank_candidates(frame, rows)
            except Exception:
                pass
        if (
            self.qd_generator is not None
            and rows
            and hasattr(self.qd_generator, "generate_candidate_pool")
        ):
            try:
                rows = self.qd_generator.generate_candidate_pool(
                    frame,
                    rows,
                    goal_energy=self.goal_bias,
                    action_effect_scorer=self.frame_change_scorer,
                    arm_label="energy-QD",
                )
            except Exception:
                self._qd_generation_errors += 1
        return self._apply_controllable_novelty_order(frame, rows)

    def _apply_amortized_prior_order(
        self,
        frame: Any,
        candidates: Sequence[Mapping[str, Any]],
        *,
        path: Sequence[Mapping[str, Any]] | None = None,
    ) -> list[dict[str, Any]]:
        """REQ-ARC-WMTE-4701: rank first-contact proposals before search consumes them."""

        if self.amortized_first_contact_prior is None:
            return [dict(row) for row in candidates]
        return self.amortized_first_contact_prior.rank_candidates(frame, candidates, path=path)

    def _apply_object_centric_proposal_order(
        self,
        frame: Any,
        candidates: Sequence[Mapping[str, Any]],
        *,
        previous_frame: Any | None = None,
    ) -> list[dict[str, Any]]:
        """REQ-ARC-WMTE-4700: augment/rank proposals with object-centric slots."""

        if self.object_centric_proposal_policy is None:
            return [dict(row) for row in candidates]
        return self.object_centric_proposal_policy.rank_candidates(
            frame,
            candidates,
            previous_frame=previous_frame,
        )

    def _apply_controllable_novelty_order(
        self,
        frame: Any,
        candidates: Sequence[Mapping[str, Any]],
    ) -> list[dict[str, Any]]:
        """REQ-ARC-WMTE-4688: apply intrinsic proposal bonus before value ranking."""

        if self.controllable_novelty_policy is None:
            return [dict(row) for row in candidates]
        return self.controllable_novelty_policy.rank_candidates(frame, candidates)

    @staticmethod
    def _game_over(frame) -> bool:
        from carnot.agentic.arc_agi3_live_adapter import _game_over

        try:
            return bool(_game_over(frame))
        except Exception:
            return False

    def _value(
        self,
        frame,
        node_hash: str | None = None,
        previous_frame: Any | None = None,
    ) -> float:
        """Frame-only learned progress score (predicted steps-to-next-level-up; LOWER == closer).
        0.0 when no value head -> the frontier falls back to shallowest-first. Never crashes the loop.
        Also 0.0 when value_weight==0: at weight 0 the value term is multiplied by 0 in the frontier
        priority (ordering is depth-primary + on-path tiebreak, identical with or without the value), so
        computing the expensive v3 featurizer per node would be pure dead cost (the 2026-06-20 regression:
        it made the weight-0 submitted default slower than bare BFS for ZERO routing benefit). The v3 head
        stays fully wired and fires unchanged whenever value_weight>0.

        Positive weights use a frame-hash cache so repeated frontier visits do
        not re-run the expensive v3 featurizer.
        """
        if self.value_head is None or self.value_weight == 0.0:
            return 0.0
        if node_hash is not None and node_hash in self._value_cache:
            self._value_cache_hits += 1
            return self._value_cache[node_hash]
        try:
            try:
                value = float(self.value_head(frame, previous_frame=previous_frame))
            except TypeError:
                value = float(self.value_head(frame))
        except Exception:
            value = 0.0
        self._value_head_evals += 1
        if node_hash is not None:
            self._value_cache[node_hash] = value
        return value

    def _initial_value(self, frame) -> tuple[float | None, Any | None]:
        if self.value_head is None or self.value_weight == 0.0:
            return 0.0, None
        return None, frame

    @staticmethod
    def _same_path_step(left: Mapping[str, Any], right: Mapping[str, Any]) -> bool:
        return int(left.get("action")) == int(right.get("action")) and left.get(
            "data"
        ) == right.get("data")

    @classmethod
    def _path_is_prefix(
        cls, prefix: Sequence[Mapping[str, Any]], path: Sequence[Mapping[str, Any]]
    ) -> bool:
        if len(prefix) > len(path):
            return False
        return all(cls._same_path_step(left, right) for left, right in zip(prefix, path))

    def _record_forward_edge(
        self, origin: Optional[str], action: Mapping[str, Any], next_hash: str
    ) -> None:
        if origin is None or next_hash == origin:
            return
        act = {"action": int(action["action"]), "data": action.get("data")}
        edges = self.adj.setdefault(origin, [])
        if any(existing == act and nxt == next_hash for existing, nxt in edges):
            return
        edges.append((act, next_hash))
        self._nav_edges_recorded += 1

    def _similarity_descriptor(self, frame: Any) -> tuple[int, ...] | None:
        """REQ-ARC-WMTE-4933: deterministic coarse key for within-game near-state lookup."""

        if frame is None:
            return None
        try:
            from carnot.agentic.arc_value_learner import cross_game_features_v2

            values = cross_game_features_v2(frame)
        except Exception:
            return None
        bucket: list[int] = []
        for value in values:
            try:
                number = float(value)
            except (TypeError, ValueError):
                number = 0.0
            if number != number:
                number = 0.0
            bucket.append(int(round(number / self.similarity_bucket_width)))
        return tuple(bucket)

    def _index_similarity_state(self, node_hash: str, frame: Any) -> None:
        """Index one live-observed state; never imports cross-game trajectories."""

        if not self.similarity_retrieval_enabled or not node_hash:
            return
        descriptor = self._similarity_descriptor(frame)
        if descriptor is None:
            return
        self._similarity_descriptor_by_hash[node_hash] = descriptor
        bucket = self._similarity_state_buckets.setdefault(descriptor, [])
        if node_hash not in bucket:
            bucket.append(node_hash)
            bucket.sort()

    def _ingest(self, latest) -> None:
        if latest is None:
            return
        h = self._hash(latest)
        lvl = _level_of(latest)
        over = self._game_over(latest)
        previous_best_level = int(self.best_level)
        initial_observation = self.start_level is None
        if self.start_level is None:
            self.start_level = lvl
        level_increased = (not initial_observation) and lvl > previous_best_level
        self.best_level = max(self.best_level, lvl)
        features = None
        if self.awaiting is not None:
            o = self.awaiting
            self.awaiting = None
            if self.dense_curiosity is not None and o.get("grid") is not None:
                try:
                    from carnot.agentic.arc_agi3_world_model import grid_of

                    self.dense_curiosity.observe_transition(
                        str(o["origin"]),
                        h,
                        o["grid"],
                        int(o["action"]),
                        o["data"],
                        grid_of(latest),
                        level_before=int(o.get("level_before") or 0),
                        level_after=int(lvl),
                    )
                except Exception:
                    pass
            if self.controllable_novelty_policy is not None and o.get("grid") is not None:
                try:
                    action = {"action": int(o["action"]), "data": o.get("data")}
                    self.controllable_novelty_policy.record_transition(
                        o.get("previous_frame") or o.get("grid"),
                        latest,
                        action,
                    )
                except Exception:
                    pass
            if self.object_centric_proposal_policy is not None and o.get("grid") is not None:
                try:
                    action = {"action": int(o["action"]), "data": o.get("data")}
                    self.object_centric_proposal_policy.record_transition(
                        o.get("previous_frame") or o.get("grid"),
                        latest,
                        action,
                    )
                except Exception:
                    pass
            # OBSERVE hook (REQ-ARC-OAE-4710): feed realized (before, action, after) triples to
            # an online scorer that supports observe_transition.  The LiveActionEffectScorer (the
            # frozen shipped scorer) does NOT have this method, so the check `hasattr(fcs, ...)`
            # makes this a guaranteed no-op for the frozen path -- byte-identical parity.
            fcs = getattr(self, "frame_change_scorer", None)
            if (
                fcs is not None
                and hasattr(fcs, "observe_transition")
                and o.get("previous_frame") is not None
            ):
                try:
                    fcs.observe_transition(
                        o.get("previous_frame") or o.get("grid"),
                        int(o["action"]),
                        o.get("data"),
                        latest,
                    )
                except Exception:
                    pass
            action_prior = getattr(self, "action_prior", None)
            if (
                action_prior is not None
                and hasattr(action_prior, "observe_transition")
                and o.get("previous_frame") is not None
            ):
                try:
                    action_prior.observe_transition(
                        o.get("previous_frame") or o.get("grid"),
                        int(o["action"]),
                        o.get("data"),
                        latest,
                    )
                except Exception:
                    pass
            if level_increased:
                fcs = getattr(self, "frame_change_scorer", None)
                if fcs is not None and hasattr(fcs, "reset"):
                    try:
                        fcs.reset(level=int(lvl), reset_to_prior=True)
                    except TypeError:
                        try:
                            fcs.reset()
                        except Exception:
                            pass
                    except Exception:
                        pass
                action_prior = getattr(self, "action_prior", None)
                if action_prior is not None and hasattr(action_prior, "reset"):
                    try:
                        action_prior.reset(level=int(lvl), reset_to_prior=True)
                    except TypeError:
                        try:
                            action_prior.reset()
                        except Exception:
                            pass
                    except Exception:
                        pass
            if over:
                origin_node = self.graph.get(o["origin"], {})
                act = {"action": o["action"], "data": o["data"]}
                new_path = list(origin_node.get("path", [])) + [act]
                self._record_discriminative_sample(
                    latest,
                    previous_frame=o.get("previous_frame") or origin_node.get("frame"),
                    label=0,
                    source="game_over",
                    node_hash=h,
                    path=new_path,
                )
            else:
                act = {"action": o["action"], "data": o["data"]}
                # record the forward edge for frontier-distance navigation (only if the
                # action actually CHANGED state — a no-op self-edge is useless to navigate)
                self._record_forward_edge(o["origin"], act, h)
                if h not in self.graph:
                    origin_node = self.graph.get(o["origin"], {})
                    opath = origin_node.get("path", [])
                    new_path = opath + [act]
                    features = self._record_discriminative_sample(
                        latest,
                        previous_frame=o.get("previous_frame") or origin_node.get("frame"),
                        label=1,
                        source="alive_frontier",
                        node_hash=h,
                        path=new_path,
                    )
                    value, frame_for_value = self._initial_value(latest)
                    self.graph[h] = {
                        "path": new_path,
                        "untested": self._candidates(
                            latest,
                            path=new_path,
                            previous_frame=o.get("previous_frame") or origin_node.get("frame"),
                        ),
                        "value": value,
                        "frame": (
                            latest
                            if self.goal_bias is not None
                            or self.dense_curiosity is not None
                            or self.action_effect_expansion_prior is not None
                            or self.qd_generator is not None
                            or self.controllable_novelty_policy is not None
                            or self.object_centric_proposal_policy is not None
                            or self.go_explore_archive is not None
                            or self.similarity_retrieval_enabled
                            else frame_for_value
                        ),
                        "previous_frame": o.get("previous_frame") or origin_node.get("frame"),
                        "discriminative_features": features,
                    }
                self._index_similarity_state(h, latest)
        self.cur = h
        if self.root is None and not over:
            self.root = h
            features = self._record_discriminative_sample(
                latest,
                label=1,
                source="root",
                node_hash=h,
                path=[],
            )
            value, frame_for_value = self._initial_value(latest)
            self.graph.setdefault(
                h,
                {
                    "path": [],
                    "untested": self._candidates(latest, path=[], previous_frame=None),
                    "value": value,
                    "frame": (
                        latest
                        if self.goal_bias is not None
                        or self.dense_curiosity is not None
                        or self.action_effect_expansion_prior is not None
                        or self.qd_generator is not None
                        or self.controllable_novelty_policy is not None
                        or self.object_centric_proposal_policy is not None
                        or self.go_explore_archive is not None
                        or self.similarity_retrieval_enabled
                        else frame_for_value
                    ),
                    "previous_frame": None,
                    "discriminative_features": features,
                },
            )
            self._index_similarity_state(h, latest)
        if self.go_explore_archive is not None and not over:
            node = self.graph.get(h)
            if node is not None:
                self.go_explore_archive.observe(latest, node.get("path") or [])

    def _frontier(self) -> Optional[str]:
        # BRIDGE: A*-style frontier order -- priority = depth + value_weight*value. value_weight=0 is
        # depth-primary (pure BFS; value only breaks ties -> provably cannot regress). value_weight>0
        # lets the value head NUDGE toward predicted-closer states (the routing that unlocked cn04 in
        # graph_explore at weight 5). A full value-OVERRIDE (ignoring depth) measurably REGRESSED the
        # baseline (the weak head misroutes from shallow wins), so the blend keeps depth load-bearing.
        use_value = self.value_head is not None and self.value_weight != 0.0
        w = self.value_weight
        eligible: list[tuple[str, dict, int, float, float, float, float]] = []
        for h, node in self.graph.items():
            if not node["untested"]:
                self._record_discriminative_features(
                    node.get("discriminative_features"),
                    label=0,
                    source="frontier_exhausted",
                    node_hash=h,
                    path=node.get("path") or [],
                )
                continue
            on_path = self._node_on_path_proba(node)
            node["on_path_proba"] = on_path
            if (
                self.online_discriminator is not None
                and on_path < self.discriminative_prune_threshold
            ):
                if node.get("discriminative_pruned") is not True:
                    self._disc_frontier_pruned += 1
                node["discriminative_pruned"] = True
                continue
            node["discriminative_pruned"] = False
            depth = len(node["path"])
            eligible.append(
                (
                    h,
                    node,
                    depth,
                    on_path,
                    self._action_effect_frontier_key(node),
                    self._goal_bias_score(node),
                    self._curiosity_score(h),
                )
            )

        if use_value and eligible:
            cheap_ranked = sorted(
                eligible,
                key=lambda item: (
                    item[2],
                    -item[3],
                    item[0],
                ),
            )[: self.lazy_value_top_k]
            for h, node, _depth, _on_path, _action_effect, _goal_bias, _curiosity in cheap_ranked:
                if node.get("value") is None:
                    node["value"] = self._value(
                        node.get("frame"),
                        node_hash=h,
                        previous_frame=node.get("previous_frame"),
                    )
                    if (
                        self.goal_bias is None
                        and self.dense_curiosity is None
                        and self.action_effect_expansion_prior is None
                        and self.qd_generator is None
                        and self.controllable_novelty_policy is None
                        and self.object_centric_proposal_policy is None
                        and self.go_explore_archive is None
                    ):
                        node["frame"] = None

        best = None
        best_key = None
        for h, node, depth, on_path, action_effect, goal_bias, curiosity in eligible:
            nav_key = self._frontier_navigation_cost_key(h) if self.navigation_cost_tiebreak else ()
            if self.navigation_cost_tiebreak and use_value:
                value = node.get("value", 0.0)
                if value is None:
                    value = 0.0
                key = (
                    depth,
                    float(action_effect),
                    w * float(value),
                    self._goal_bias_key(goal_bias),
                    -float(curiosity),
                    *nav_key,
                    -on_path,
                )
            elif use_value:
                value = node.get("value", 0.0)
                if value is None:
                    value = 0.0
                key = (
                    depth + w * float(value),
                    depth,
                    float(action_effect),
                    self._goal_bias_key(goal_bias),
                    -float(curiosity),
                    -on_path,
                )
            elif self.navigation_cost_tiebreak:
                key = (
                    depth,
                    float(action_effect),
                    self._goal_bias_key(goal_bias),
                    -float(curiosity),
                    *nav_key,
                    -on_path,
                )
            else:
                key = (
                    depth,
                    float(action_effect),
                    self._goal_bias_key(goal_bias),
                    -float(curiosity),
                    -on_path,
                )
            if best is None or key < best_key:
                best, best_key = h, key
        return best

    def _frontier_navigation_cost_key(self, node_hash: str) -> tuple[int, int]:
        """SCENARIO-ARC-FCP-4523: prefer cheap navigation only within equal depth."""

        fwd = self._shortest_path(self.cur, node_hash, allow_similarity=False)
        if fwd is not None:
            return (0, len(fwd))
        return (1, len(self.graph.get(node_hash, {}).get("path", [])))

    def _exact_shortest_path(self, src: Optional[str], dst: str) -> Optional[list]:
        """Frontier-distance navigation: BFS over the KNOWN forward edges from src to dst.
        Returns the action sequence to walk there WITHOUT a RESET (cheaper than replay-
        from-root), or None if dst isn't forward-reachable from src in the known graph."""
        from collections import deque

        if src is None or src == dst:
            return [] if src == dst else None
        seen = {src}
        q = deque([(src, [])])
        while q:
            node, path = q.popleft()
            for act, nxt in self.adj.get(node, []):
                if nxt in seen:
                    continue
                npath = path + [act]
                if nxt == dst:
                    return npath
                seen.add(nxt)
                q.append((nxt, npath))
        return None

    def _shortest_path(
        self,
        src: Optional[str],
        dst: str,
        *,
        allow_similarity: bool = True,
    ) -> Optional[list]:
        exact = self._exact_shortest_path(src, dst)
        if exact is not None:
            self._last_shortest_path_kind = "exact"
            return exact
        similar = self._similarity_shortest_path(src, dst) if allow_similarity else None
        self._last_shortest_path_kind = "similarity" if similar is not None else None
        return similar

    def _edge_next_hash(self, origin: str, action: Mapping[str, Any]) -> str | None:
        act = {"action": int(action["action"]), "data": action.get("data")}
        for existing, next_hash in self.adj.get(origin, []):
            if self._same_path_step(existing, act):
                return next_hash
        return None

    @staticmethod
    def _similarity_world_model_key(grid: Any, action: int, data: Any) -> tuple:
        import numpy as np

        arr = np.asarray(grid, dtype=np.int16)
        data_key = json.dumps(data, sort_keys=True, separators=(",", ":"), default=str)
        return (tuple(int(dim) for dim in arr.shape), arr.tobytes(), int(action), data_key)

    def _similarity_world_model_verifier_passes(
        self,
        origin: str,
        dst: str,
        prefix: Sequence[Mapping[str, Any]],
    ) -> bool:
        try:
            import numpy as np

            from carnot.agentic.arc_agi3_world_model import grid_of
            from carnot.agentic.arc_executable_world_model import Transition, WorldModelVerifier
        except Exception:
            return False

        transitions = []
        lookup: dict[tuple, Any] = {}
        node_hash = origin
        for step in prefix:
            next_hash = self._edge_next_hash(node_hash, step)
            if next_hash is None:
                return False
            before = self.graph.get(node_hash, {}).get("frame")
            after = self.graph.get(next_hash, {}).get("frame")
            if before is None or after is None:
                return False
            try:
                grid = np.asarray(grid_of(before), dtype=np.int16)
                next_grid = np.asarray(grid_of(after), dtype=np.int16)
            except Exception:
                return False
            action = int(step["action"])
            data = step.get("data")
            lookup[self._similarity_world_model_key(grid, action, data)] = next_grid.copy()
            transitions.append(
                Transition(
                    grid=grid.copy(),
                    action=action,
                    data=data,
                    next_grid=next_grid.copy(),
                    level_before=0,
                    level_after=0,
                )
            )
            node_hash = next_hash
        if node_hash != dst or not transitions:
            return False

        def engine(grid: Any, action: int, data: Any) -> Any:
            key = self._similarity_world_model_key(grid, int(action), data)
            if key not in lookup:
                raise KeyError("unobserved_similarity_prefix_transition")
            return lookup[key].copy()

        try:
            score = WorldModelVerifier(transitions).score(engine)
            self._nav_similarity_world_model_verifier_checks += 1
            return float(score.accuracy) >= 1.0
        except Exception:
            return False

    def _similarity_prefix_router_accepts(
        self,
        src: str,
        origin: str,
        dst: str,
        prefix: Sequence[Mapping[str, Any]],
    ) -> bool:
        src_node = self.graph.get(src) or {}
        dst_node = self.graph.get(dst) or {}
        src_frame = src_node.get("frame")
        dst_frame = dst_node.get("frame")
        if src_frame is None or dst_frame is None:
            return False
        if self.value_head is not None and self.value_weight != 0.0:
            self._nav_similarity_value_checks += 1
            src_value = self._value(src_frame, node_hash=src)
            dst_value = self._value(dst_frame, node_hash=dst)
            if float(dst_value) > float(src_value):
                return False
        if self.goal_bias is not None:
            self._nav_similarity_goal_checks += 1
            src_goal = self._goal_bias_score(src_node)
            dst_goal = self._goal_bias_score(dst_node)
            if self._goal_bias_key(dst_goal) > self._goal_bias_key(src_goal):
                return False
        return self._similarity_world_model_verifier_passes(origin, dst, prefix)

    def _similarity_shortest_path(self, src: Optional[str], dst: str) -> Optional[list]:
        if not self.similarity_retrieval_enabled or src is None or src == dst:
            return None
        descriptor = self._similarity_descriptor_by_hash.get(src)
        if descriptor is None:
            return None
        candidates = [
            state_hash
            for state_hash in self._similarity_state_buckets.get(descriptor, [])
            if state_hash != src
        ][: self.similarity_max_candidates]
        for state_hash in candidates:
            prefix = self._exact_shortest_path(state_hash, dst)
            if not prefix:
                continue
            self._nav_similarity_candidates_considered += 1
            if self._similarity_prefix_router_accepts(src, state_hash, dst, prefix):
                self._nav_similarity_router_accepts += 1
                return prefix
            self._nav_similarity_router_rejects += 1
        return None

    def _partial_forward_path(self, src: Optional[str], dst: str) -> Optional[list]:
        """Walk to the deepest reachable ancestor of dst, then replay only the suffix."""

        if src is None or dst not in self.graph:
            return None
        target_path = self.graph.get(dst, {}).get("path", [])
        best_depth = -1
        best_plan = None
        for ancestor, node in self.graph.items():
            if ancestor == dst:
                continue
            ancestor_path = node.get("path", [])
            if not self._path_is_prefix(ancestor_path, target_path):
                continue
            forward = self._exact_shortest_path(src, ancestor)
            if forward is None:
                continue
            depth = len(ancestor_path)
            if depth <= best_depth:
                continue
            suffix = [
                {"action": int(step["action"]), "data": step.get("data")}
                for step in target_path[depth:]
            ]
            best_depth = depth
            best_plan = list(forward) + suffix
        return best_plan

    def _serve(self) -> tuple:
        item = self.pending.pop(0)
        if item["kind"] == "RESET":
            self.awaiting = None  # RESET has no forward edge to attribute
            return ("RESET", None)
        if item.get("probe"):
            origin = item.get("origin", self.cur)
            if "origin" not in item:
                self._drop_queued_action_from_current_frontier(origin, item)
            self.awaiting = {
                "origin": origin,
                "action": item["kind"],
                "data": item["data"],
                "grid": self._grid_for_hash(origin),
                "level_before": int(self.best_level),
                "previous_frame": self.graph.get(origin, {}).get("frame"),
            }
        else:
            # nav / RESET-replay step (probe:False): attribute its forward edge from the CURRENT state so
            # adj FILLS IN the replayed path. Previously only probe steps recorded edges, so replayed paths
            # were never learned -> _shortest_path returned None -> every backtrack RESET-replayed from root,
            # burning actions (the 2026-06-20 regression: lp85 7792 actions vs bare BFS's 21). Recording
            # these edges lets future navigation use _shortest_path (forward-walk) instead of RESET-replay.
            self.awaiting = {
                "origin": self.cur,
                "action": item["kind"],
                "data": item["data"],
                "grid": self._grid_for_hash(self.cur),
                "level_before": int(self.best_level),
                "previous_frame": self.graph.get(self.cur, {}).get("frame"),
            }
        return (item["kind"], item["data"])

    def _drop_queued_action_from_current_frontier(
        self, origin: Optional[str], item: Mapping[str, Any]
    ) -> None:
        if origin is None:
            return
        node = self.graph.get(origin)
        if not node:
            return
        act = {"action": int(item["kind"]), "data": item.get("data")}
        for idx, candidate in enumerate(list(node.get("untested") or [])):
            if self._same_path_step(candidate, act):
                del node["untested"][idx]
                return

    def _pop_frontier_batch(self, node: dict) -> list[dict]:
        limit = (
            len(node["untested"]) if self.frontier_batch_size is None else self.frontier_batch_size
        )
        count = min(int(limit), len(node["untested"]))
        actions = node["untested"][:count]
        del node["untested"][:count]
        return actions

    def _pop_untested(self, node):
        """Pop the next untested action: the most-salient (pop(0)) normally; but when hybrid diversity is on
        AND the search has STALLED (no new level for _stall_threshold moves), pop a RANDOM one among the
        top-K -- the injection that recovers the structure-missed wins (r11l/sp80) the depth-first ride over-
        commits past. Flag OFF -> always pop(0) -> byte-identical to the submitted behavior."""
        lst = node["untested"]
        if (
            self._hybrid_diversity
            and self._steps_since_progress > self._stall_threshold
            and len(lst) > 1
        ):
            return lst.pop(self._div_rng.randrange(min(len(lst), self._div_topk)))
        return lst.pop(0)

    def _qd_sequence_for_node(self, node: Mapping[str, Any]) -> list[dict[str, Any]]:
        """REQ-ARC-WMTE-4653: generate one additive multi-action sequence for a frontier node."""

        if self.qd_generator is None or node.get("qd_sequence_injected"):
            return []
        frame = node.get("frame")
        candidates = list(node.get("untested") or [])
        if frame is None or not candidates:
            return []
        try:
            sequence = self.qd_generator.best_sequence(
                frame,
                candidates,
                goal_energy=self.goal_bias,
                action_effect_scorer=self.frame_change_scorer,
                min_len=2,
            )
        except Exception:
            self._qd_generation_errors += 1
            return []
        rows = [
            {"action": int(step["action"]), "data": step.get("data")}
            for step in sequence
            if step.get("action") is not None
        ]
        if len(rows) < 2:
            return []
        if isinstance(node, dict):
            node["qd_sequence_injected"] = True
        self._qd_sequences_injected += 1
        self._qd_actions_injected += len(rows)
        return rows

    def _begin_qd_sequence(self, sequence: Sequence[Mapping[str, Any]]) -> tuple:
        first = sequence[0]
        rest = sequence[1:]
        self.pending = [
            {"kind": int(step["action"]), "data": step.get("data"), "probe": True} for step in rest
        ]
        self.awaiting = {
            "origin": self.cur,
            "action": int(first["action"]),
            "data": first.get("data"),
            "grid": self._grid_for_hash(self.cur),
            "level_before": int(self.best_level),
            "previous_frame": self.graph.get(self.cur, {}).get("frame"),
        }
        return (int(first["action"]), first.get("data"))

    def _go_explore_replay_sequence(
        self,
        *,
        current_path: Sequence[Mapping[str, Any]] | None = None,
    ) -> list[dict[str, Any]]:
        """REQ-ARC-WMTE-4701: ask the archive for a reset-replayable return prefix."""

        if self.go_explore_archive is None:
            return []
        return self.go_explore_archive.select_prefix(current_path=current_path)

    def _begin_go_explore_replay(self, sequence: Sequence[Mapping[str, Any]]) -> tuple:
        self._go_explore_prefixes_injected += 1
        self._go_explore_actions_injected += len(sequence)
        self.pending = [{"kind": "RESET", "data": None, "probe": False}]
        self.pending += [
            {"kind": int(step["action"]), "data": step.get("data"), "probe": False}
            for step in sequence
        ]
        return self._serve()

    def next_move(self, frames, latest) -> tuple:
        if self.root is None and latest is None:  # bootstrap: RESET to get the first frame
            return ("RESET", None)
        self._ingest(latest)
        if self._hybrid_diversity and latest is not None:  # track stall for the diversity injection
            _lvl = _level_of(latest)
            if _lvl > self._nm_best_level:
                self._nm_best_level = _lvl
                self._steps_since_progress = 0
            else:
                self._steps_since_progress += 1
        if self.pending:
            return self._serve()
        over = latest is not None and self._game_over(latest)
        cur_node = self.graph.get(self.cur) if not over else None
        if (
            self.go_explore_archive is not None
            and cur_node
            and (not cur_node["untested"] or len(cur_node["path"]) >= self.max_depth)
        ):
            replay = self._go_explore_replay_sequence(current_path=cur_node.get("path") or [])
            if replay:
                return self._begin_go_explore_replay(replay)
        # 1) DEPTH-first ride (search_mode="depth_first_ride", default): expand the current state's
        #    untested SALIENT actions while under the depth cap (no nav cost; reaches the deep wins
        #    lp85/sp80 need — BFS-order regressed those). best_first SKIPS this and always expands the
        #    globally-best A*-value frontier (step 2) so the value head drives the search order.
        if (
            self.search_mode == "depth_first_ride"
            and cur_node
            and cur_node["untested"]
            and len(cur_node["path"]) < self.max_depth
        ):
            qd_sequence = self._qd_sequence_for_node(cur_node)
            if qd_sequence:
                return self._begin_qd_sequence(qd_sequence)
            a = self._pop_untested(cur_node)
            self.awaiting = {
                "origin": self.cur,
                "action": a["action"],
                "data": a["data"],
                "grid": self._grid_for_hash(self.cur),
                "level_before": int(self.best_level),
                "previous_frame": self.graph.get(self.cur, {}).get("frame"),
            }
            return (a["action"], a["data"])
        # 2) Expand the best frontier (A*-value order). In best_first this is the primary step; in
        #    depth_first_ride it fires when the current node is exhausted / dead-end / depth-capped.
        th = self._frontier()
        if th is None:
            self.explored_out = True
            return (None, None)
        node = self.graph[th]
        if (
            th == self.cur and not over
        ):  # best frontier IS the current state -> expand in place (no nav)
            qd_sequence = self._qd_sequence_for_node(node)
            if qd_sequence:
                return self._begin_qd_sequence(qd_sequence)
            a = self._pop_untested(node)
            self.awaiting = {
                "origin": self.cur,
                "action": a["action"],
                "data": a["data"],
                "grid": self._grid_for_hash(self.cur),
                "level_before": int(self.best_level),
                "previous_frame": self.graph.get(self.cur, {}).get("frame"),
            }
            return (a["action"], a["data"])
        batch = self._pop_frontier_batch(node)
        self._nav_attempts += 1
        fwd = self._shortest_path(self.cur, th) if not over else None
        if fwd is not None:
            if self._last_shortest_path_kind == "similarity":
                self._nav_similarity_hits += 1
            else:
                self._nav_exact_hits += 1
            self._nav_forward_steps += len(fwd)
            self.pending = [{"kind": s["action"], "data": s["data"], "probe": False} for s in fwd]
        else:
            partial = self._partial_forward_path(self.cur, th) if not over else None
            if partial is not None:
                self._nav_partial_hits += 1
                self._nav_forward_steps += len(partial)
                self.pending = [
                    {"kind": s["action"], "data": s["data"], "probe": False} for s in partial
                ]
            else:
                self._nav_reset_fallbacks += 1
                self._nav_reset_replay_steps += len(node["path"])
                self.pending = [{"kind": "RESET", "data": None, "probe": False}]
                self.pending += [
                    {"kind": s["action"], "data": s["data"], "probe": False} for s in node["path"]
                ]
        for idx, action in enumerate(batch):
            item = {"kind": action["action"], "data": action["data"], "probe": True}
            if idx == 0:
                item["origin"] = th
            self.pending.append(item)
        return self._serve()

    def is_done(self, frames, latest) -> bool:
        if latest is not None:
            lvl = _level_of(latest)
            if self.start_level is None:
                self.start_level = lvl
            self.best_level = max(self.best_level, lvl)
        if (
            self.start_level is not None
            and self.best_level >= self.start_level + self.target_levels
        ):
            return True
        # Smart grace-period early-stop: cut the fruitless post-solve tail WITHOUT capping levels. Once at
        # least one level is reached, a new level-up resets the window; if no new level appears within
        # `early_stop_grace` moves, stop (the agent is churning on unreachable deeper levels, and every
        # extra action quadratically erodes the (human/agent_actions)^2 efficiency score). Riding
        # consecutive level-ups keeps the window alive, so reachable deeper levels are still solved.
        if (
            self.early_stop_grace is not None
            and self.start_level is not None
            and self.best_level > self.start_level
        ):
            if self.best_level > self._early_stop_level_mark:
                self._early_stop_level_mark = self.best_level
                self._early_stop_frame_mark = len(frames)
            elif (len(frames) - self._early_stop_frame_mark) > self.early_stop_grace:
                self.early_stopped = True
                return True
        return self.explored_out


def _value_routing_feature_indices() -> list[int]:
    """REQ-LEARN-4652: full-v3 indices kept by the cheap live value route."""

    from carnot.agentic.arc_value_learner import cross_game_feature_slices_v3

    slices = cross_game_feature_slices_v3()
    out: list[int] = []
    for name in ("v2", "frame_delta"):
        start, stop = slices[name]
        out.extend(range(start, stop))
    return out


class _SlicedLinearValueHead:
    """Linear value head over the REQ-LEARN-4652 v2+frame-delta subset."""

    feature_subset = SUBMITTED_VALUE_HEAD_FEATURE_SUBSET
    verifier_is_oracle = False

    def __init__(self, weights: Sequence[float], bias: float) -> None:
        self.weights = [float(value) for value in weights]
        self.bias = float(bias)

    def __call__(self, frame: Any, previous_frame: Any | None = None) -> float:
        from carnot.agentic.arc_value_learner import cross_game_features_v3_value_routing

        features = cross_game_features_v3_value_routing(frame, previous_frame=previous_frame)
        if len(features) != len(self.weights):
            return 0.0
        value = sum(weight * float(feature) for weight, feature in zip(self.weights, features))
        return float(max(0.0, value + self.bias))


def _load_sliced_v3_value_head(path: Path) -> _SlicedLinearValueHead | None:
    from carnot.agentic.arc_value_learner import cross_game_feature_slices_v3

    payload = json.loads(path.read_text(encoding="utf-8"))
    weights = [float(value) for value in payload.get("weights") or []]
    if not weights:
        return None
    indices = _value_routing_feature_indices()
    full_width = max(stop for _start, stop in cross_game_feature_slices_v3().values())
    if len(weights) == full_width + 1:
        return _SlicedLinearValueHead([weights[index] for index in indices], weights[-1])
    if len(weights) == len(indices) + 1:
        return _SlicedLinearValueHead(weights[:-1], weights[-1])
    return None


def _load_linear_cross_game_value_head(root: Path | str = REPO):
    """Legacy linear value-head loader, with REQ-LEARN-4652 cheap-v3 slicing first."""

    models = Path(root) / "models"
    try:
        from carnot.agentic.arc_value_learner import (
            DaggerWinReachabilityValueHead,
            LearnedVerifier,
            cross_game_features,
            cross_game_features_v2,
        )

        dagger = Path(root) / DAGGER_VALUE_HEAD_RELATIVE_PATH
        if dagger.exists():
            return DaggerWinReachabilityValueHead.load(dagger)
        v3 = models / "arc_verifier_cross_game_v3.json"
        if v3.exists():
            sliced = _load_sliced_v3_value_head(v3)
            if sliced is not None:
                return sliced
        # prefer the RICHER v2 head (spatial occupancy; it routed cn04 where v1's 5 scalars could not)
        v2 = models / "arc_verifier_cross_game_v2.json"
        if v2.exists():
            v = LearnedVerifier.load(v2, cross_game_features_v2)
            return lambda frame: v(frame)
        v1 = models / "arc_verifier_cross_game.json"
        if v1.exists():
            v = LearnedVerifier.load(v1, cross_game_features)
            return lambda frame: v(frame)
    except Exception:
        return None
    return None


def load_cross_game_value_head():
    """BRIDGE loader for the frame-only cross-game value head.

    Prefer the graduated position-preserving SpatialValueNet when a live checkpoint exists. If no
    spatial checkpoint is available, fall back to the legacy linear checkpoint so older local setups keep
    running and experiment 4617 can measure that baseline explicitly.
    """

    spatial = load_live_spatial_value_head()
    if spatial is not None:
        return spatial
    return _load_linear_cross_game_value_head()


class CarnotAgentPolicy:
    """Framework-agnostic decision logic. `next_move` yields ("RESET",None) once, then
    the banked plan one step at a time, then (None,None) when exhausted. `is_done`
    stops when the target level is reached or the plan is spent."""

    def __init__(
        self,
        game_id: str,
        solutions: Optional[dict] = None,
        target_level: Optional[int] = None,
        force_explore: bool = False,
        hud_mask=None,
        value_head=None,
        value_weight: float = 0.0,
        search_mode: str = "depth_first_ride",
        frame_change_scorer: Any | None = None,
        frame_change_prune_threshold: float | None = None,
        action_effect_expansion_prior: Any | bool | None = None,
        action_prior: Any | None = None,
        action_prior_prune_quantile: float | None = None,
        adaptive_budget_threshold: float | None = None,
        adaptive_budget_value_head: Any | None = None,
        adaptive_budget_noop_threshold: float = 0.5,
        lazy_value_top_k: int = SUBMITTED_LAZY_VALUE_TOP_K,
        frontier_batch_size: int | str | None = SUBMITTED_FRONTIER_BATCH_SIZE,
        navigation_cost_tiebreak: bool = SUBMITTED_NAVIGATION_COST_TIEBREAK,
        candidate_router: Any | None = None,
        similarity_retrieval: bool | None = None,
    ) -> None:
        self.short = str(game_id).split("-", 1)[0]
        sols = solutions if solutions is not None else load_solutions()
        self.plan = [] if force_explore else sols.get(self.short, [])
        self.i = 0
        self.reset_sent = False
        self.target = target_level if target_level is not None else CLAIMED.get(self.short, 1)
        self.has_plan = bool(self.plan)
        # eval games are UNSEEN -> no banked plan -> the generic step-wise explorer runs (value_head +
        # value_weight A*-route its frontier when provided -- the offline->live bridge).
        self.explorer: Optional[StepwiseExplorer] = (
            None
            if self.has_plan
            else StepwiseExplorer(
                hud_mask=hud_mask,
                value_head=value_head,
                value_weight=value_weight,
                search_mode=search_mode,
                frame_change_scorer=frame_change_scorer,
                frame_change_prune_threshold=frame_change_prune_threshold,
                action_effect_expansion_prior=action_effect_expansion_prior,
                action_prior=action_prior,
                action_prior_prune_quantile=action_prior_prune_quantile,
                adaptive_budget_threshold=adaptive_budget_threshold,
                adaptive_budget_value_head=adaptive_budget_value_head,
                adaptive_budget_noop_threshold=adaptive_budget_noop_threshold,
                lazy_value_top_k=lazy_value_top_k,
                frontier_batch_size=frontier_batch_size,
                navigation_cost_tiebreak=navigation_cost_tiebreak,
                candidate_router=candidate_router,
                similarity_retrieval=similarity_retrieval,
            )
        )

    def next_move(self, frames, latest_frame) -> tuple:
        """-> ("RESET", None) | (action_id:int, data:dict|None) | (None, None)."""
        if self.explorer is not None:  # unknown game: generic solver
            return self.explorer.next_move(frames, latest_frame)
        if not self.reset_sent:  # known game: replay banked solution
            self.reset_sent = True
            return ("RESET", None)
        if self.i < len(self.plan):
            s = self.plan[self.i]
            self.i += 1
            return (int(s["action"]), s.get("data"))
        return (None, None)

    def is_done(self, frames, latest_frame) -> bool:
        if self.explorer is not None:
            return self.explorer.is_done(frames, latest_frame)
        if _level_of(latest_frame) >= self.target:
            return True
        return self.reset_sent and self.i >= len(self.plan)


class E3AgentPolicy:
    """E3-mode agent: the STRONG choose_action. Phase machine driven step-wise by the
    harness — EXPLORE (collect transitions from its own play) -> INDUCE (an OFFLINE
    local proposer writes a world model from those transitions) -> VERIFY (Carnot
    WorldModelVerifier grounds it) -> PLAN (search to a win INSIDE the verified model)
    -> EXECUTE (replay the plan; on divergence, back to EXPLORE). The proposer is
    INJECTED and defaults to the offline-legal local one, never a closed online API, so
    this is competition-legal (no internet at eval) AND decentralized.

    The induce/verify/plan quality (esp. from a small LOCAL model) is the open milestone
    the focused loop measures next; the EXPLORE + collect + verify wiring is exercised
    here, and EXECUTE re-uses the same env interface as the explorer."""

    def __init__(
        self,
        game_id: str,
        proposer=None,
        explore_budget: Optional[int] = None,
        target_levels: int = SUBMITTED_TARGET_LEVELS,
        value_head: Any = _DEFAULT_VALUE_HEAD,
        value_weight: float = SUBMITTED_VALUE_WEIGHT,
        search_mode: str = SUBMITTED_SEARCH_MODE,
        mechanic_detector=None,
        frame_change_scorer: Any = _DEFAULT_FRAME_CHANGE_SCORER,
        frame_change_prune_threshold: float | None = None,
        action_effect_expansion_prior: Any | bool | None = None,
        action_prior: Any | None = None,
        action_prior_prune_quantile: float | None = None,
        adaptive_budget_threshold: float | None = None,
        adaptive_budget_value_head: Any | None = None,
        adaptive_budget_noop_threshold: float = 0.5,
        lazy_value_top_k: int = SUBMITTED_LAZY_VALUE_TOP_K,
        frontier_batch_size: int | str | None = SUBMITTED_FRONTIER_BATCH_SIZE,
        navigation_cost_tiebreak: bool = SUBMITTED_NAVIGATION_COST_TIEBREAK,
        candidate_router: Any = _DEFAULT_CANDIDATE_ROUTER,
        dense_curiosity: bool | DenseCuriosityProgress = False,
        dense_curiosity_weight: float = 0.15,
        dense_curiosity_discount: float = 0.5,
        goal_bias: Any = _DEFAULT_GOAL_BIAS,
        goal_candidate_guidance: Any | bool | None = (
            SUBMITTED_GOAL_ENERGY_CANDIDATE_GUIDANCE_ENABLED
        ),
        qd_generator: Any | bool | None = None,
        controllable_novelty: Any | bool | None = SUBMITTED_CONTROLLABLE_NOVELTY_PROPOSAL_ENABLED,
        object_centric_proposal: Any | bool | None = SUBMITTED_OBJECT_CENTRIC_PROPOSAL_ENABLED,
        program_synthesis_filter: Any
        | bool
        | None = SUBMITTED_PROGRAM_SYNTHESIS_PROPOSAL_FILTER_ENABLED,
        program_synthesis_filter_trust_threshold: float = (
            SUBMITTED_PROGRAM_SYNTHESIS_PROPOSAL_FILTER_TRUST_THRESHOLD
        ),
        amortized_first_contact_prior: Any | bool | None = (
            SUBMITTED_AMORTIZED_FIRST_CONTACT_PRIOR_ENABLED
        ),
        go_explore_archive: Any | bool | None = SUBMITTED_GO_EXPLORE_ARCHIVE_ENABLED,
        similarity_retrieval: bool | None = SUBMITTED_MATM_SIMILARITY_RETRIEVAL_ENABLED,
        subgoal_search: bool = False,
        subgoal_budget: int = 3,
        factored_planner: bool = False,
        factored_trust_threshold: float = 0.75,
        active_probe_controller: bool | None = None,
        active_probe_budget: int = 2,
        active_probe_concentration_threshold: float = 0.9,
        goal_guidance_lambda: float = SUBMITTED_GOAL_GUIDANCE_LAMBDA,
    ) -> None:
        import os

        self.short = str(game_id).split("-", 1)[0]
        self.target_levels = int(target_levels)
        self.goal_guidance_lambda = max(0.0, float(goal_guidance_lambda))
        if active_probe_controller is None:
            active_probe_controller = os.environ.get("CARNOT_ARC_ACTIVE_PROBE") == "1"
        self.active_probe_controller_enabled = bool(active_probe_controller)
        self.active_probe_budget = max(0, int(active_probe_budget))
        self.active_probe_concentration_threshold = max(
            0.0,
            min(1.0, float(active_probe_concentration_threshold)),
        )
        if value_head is _DEFAULT_VALUE_HEAD:
            value_head = load_cross_game_value_head()
        self.value_head = value_head
        self.subgoal_search = bool(subgoal_search)
        self.subgoal_budget = max(1, int(subgoal_budget))
        self.factored_planner = bool(factored_planner)
        self.factored_trust_threshold = max(0.0, min(1.0, float(factored_trust_threshold)))
        self.program_synthesis_filter_enabled = bool(program_synthesis_filter)
        self.program_synthesis_filter_trust_threshold = max(
            0.0,
            min(1.0, float(program_synthesis_filter_trust_threshold)),
        )
        initial_program_filter = coerce_program_synthesis_filter(program_synthesis_filter)
        if candidate_router is _DEFAULT_CANDIDATE_ROUTER:
            candidate_router = _load_submitted_candidate_router()
        if frame_change_scorer is _DEFAULT_FRAME_CHANGE_SCORER:
            frame_change_scorer = _load_submitted_frame_change_scorer()
        if action_prior is None and SUBMITTED_COLOR_BLOB_SALIENCE_ENABLED:
            action_prior = ColorBlobSaliencePrior()
        if action_effect_expansion_prior is None:
            action_effect_expansion_prior = bool(
                SUBMITTED_ACTION_EFFECT_EXPANSION_PRIOR_ENABLED and frame_change_scorer is not None
            )
        if goal_bias is _DEFAULT_GOAL_BIAS:
            goal_bias = _load_submitted_goal_energy_bias()
        self.approach_recommendation = _recommend_live_approach(self.short)
        self.strategy_route = dict(
            self.approach_recommendation.get("strategy")
            or arc_strategy_router.route_for_game(self.short)
        )
        self.approach_recommendation["strategy"] = self.strategy_route
        self._route_from_frame_checked = False
        self.mechanic_detector = mechanic_detector or _frame_only_mechanic_hint
        self.dsl_model = ObjectDeltaModel(self.short)
        self.dsl_energy: Optional[dict[str, Any]] = None
        self._dsl_transitions: list[tuple[Any, tuple, Any]] = []
        if goal_candidate_guidance is True and goal_bias is not None:
            goal_candidate_guidance = arc_goal_energy_live.GoalEnergyCandidateGuidance(
                goal_energy=goal_bias,
                transition_predictor=self._predict_goal_candidate_state,
                alpha=SUBMITTED_GOAL_ENERGY_CANDIDATE_GUIDANCE_ALPHA,
                beta=SUBMITTED_GOAL_ENERGY_CANDIDATE_GUIDANCE_BETA,
            )
        elif goal_candidate_guidance is False:
            goal_candidate_guidance = None
        self.explorer = StepwiseExplorer(
            target_levels=target_levels,
            value_head=value_head,
            value_weight=value_weight,
            search_mode=search_mode,
            frame_change_scorer=frame_change_scorer,
            frame_change_prune_threshold=frame_change_prune_threshold,
            action_effect_expansion_prior=action_effect_expansion_prior,
            action_prior=action_prior,
            action_prior_prune_quantile=action_prior_prune_quantile,
            adaptive_budget_threshold=adaptive_budget_threshold,
            adaptive_budget_value_head=adaptive_budget_value_head,
            adaptive_budget_noop_threshold=adaptive_budget_noop_threshold,
            lazy_value_top_k=lazy_value_top_k,
            frontier_batch_size=frontier_batch_size,
            navigation_cost_tiebreak=navigation_cost_tiebreak,
            candidate_router=candidate_router,
            dense_curiosity=(
                DenseCuriosityProgress(
                    self.short,
                    bonus_weight=dense_curiosity_weight,
                    backup_discount=dense_curiosity_discount,
                )
                if dense_curiosity is True
                else dense_curiosity
            ),
            dense_curiosity_weight=dense_curiosity_weight,
            dense_curiosity_discount=dense_curiosity_discount,
            goal_bias=goal_bias,
            goal_bias_label=GOAL_ENERGY_SOURCE if goal_bias is not None else "",
            goal_bias_lower_is_better=True,
            goal_candidate_guidance=goal_candidate_guidance,
            qd_generator=qd_generator,
            controllable_novelty=controllable_novelty,
            object_centric_proposal=object_centric_proposal,
            program_synthesis_filter=initial_program_filter,
            amortized_first_contact_prior=amortized_first_contact_prior,
            go_explore_archive=go_explore_archive,
            similarity_retrieval=similarity_retrieval,
        )
        self.transitions: list = []  # (grid_before, action, data, grid_after) self-collected
        self.explore_budget = (
            int(explore_budget)
            if explore_budget is not None
            else _route_explore_budget(self.strategy_route)
        )
        self.proposer = proposer  # default set lazily to LocalGGUFProposer
        self.phase = "explore"
        self.plan: list = []
        self.pi = 0
        self._prev = None  # last (grid, action_id, data) for transition pairing
        self.cell = 1
        self.induced = False
        self.root_grid = None  # the reset-state logical grid; plan_in_model starts here
        self.world_model_trust_selection = None
        self._active_probe_controller: Any = None
        self._active_probe_pending: Any = None
        self.active_probe_diagnostics: dict[str, Any] = {
            "hypothesis_posterior_built": False,
            "probe_actions_taken": 0,
            "posterior_entropy_reduction": 0.0,
            "trace": [],
            "verifier_is_oracle": False,
        }
        self._observed_level: Optional[int] = None
        self._start_level: Optional[int] = None
        self._current_goal_level: Optional[int] = None
        self._previous_level_complete_grid: Any = None
        self._episode_transition_start = 0
        self._episode_dsl_transition_start = 0
        self._level_reinduction_pending = False
        self._pending_induction_reason: Optional[str] = None
        self._execute_plan_from_current = False
        self.level_induction_events: list[dict[str, Any]] = []
        self.induction_attempts: list[dict[str, Any]] = []

    def _predict_goal_candidate_state(self, frame: Any, candidate: Mapping[str, Any]) -> Any:
        """REQ-ARC-WMTE-4737: predict a live candidate state for proposal guidance.

        This uses the agent's learned object-delta transition model. An unfitted
        model usually predicts no-op states, which the guidance detects as
        degenerate and leaves in baseline order.
        """

        from types import SimpleNamespace

        import numpy as np

        from carnot.agentic.arc_agi3_world_model import grid_of
        from carnot.agentic.arc_executable_world_model import to_logical

        data = candidate.get("data")
        action = int(candidate.get("action", candidate.get("action_id", 0)) or 0)
        grid = to_logical(grid_of(frame), self.cell)
        pred = self.dsl_model.predict(grid, _action_key(action, data))
        arr = np.asarray(pred, dtype=np.int16)
        if self.cell > 1:
            arr = np.repeat(np.repeat(arr, self.cell, axis=0), self.cell, axis=1)
        return SimpleNamespace(
            frame=arr,
            available_actions=list(getattr(frame, "available_actions", []) or []),
            levels_completed=getattr(frame, "levels_completed", 0),
        )

    def _maybe_route_from_frame(self, latest: Any) -> None:
        if self._route_from_frame_checked or latest is None:
            return
        self._route_from_frame_checked = True
        try:
            mechanic = self.mechanic_detector(latest) if self.mechanic_detector else None
        except Exception:
            mechanic = None
        if isinstance(mechanic, dict):
            mechanic = mechanic.get("mechanic")
        if not mechanic or mechanic == "unknown":
            return
        recommendation = _recommend_live_approach(self.short, mechanic=str(mechanic))
        routed = dict(
            recommendation.get("strategy")
            or arc_strategy_router.route_for_game(
                self.short,
                mechanic=str(mechanic),
            )
        )
        if routed.get("name") != self.strategy_route.get("name"):
            self.strategy_route = routed
            recommendation["strategy"] = routed
            self.approach_recommendation = recommendation
            self.explore_budget = min(self.explore_budget, _route_explore_budget(routed))

    def _fit_dsl_model(self) -> None:
        active = self._active_dsl_transitions()
        if not active:
            return
        try:
            self.dsl_model = ObjectDeltaModel(self.short).fit(active)
            self.dsl_energy = self.dsl_model.consistency_energy(active)
        except Exception:
            self.dsl_energy = {"energy": None, "n_heldout": len(active)}

    def _active_transitions(self) -> list:
        return list(self.transitions[self._episode_transition_start :])

    def _active_dsl_transitions(self) -> list[tuple[Any, tuple, Any]]:
        return list(self._dsl_transitions[self._episode_dsl_transition_start :])

    def _observe_level_boundary(self, latest: Any, *, frames_seen: int) -> list[dict[str, Any]]:
        """SCENARIO-ARC-WMTE-4533: level-up starts a new goal-acquisition episode."""

        if latest is None:
            return []
        level = _level_of(latest)
        if self._start_level is None:
            self._start_level = level
        if self._observed_level is None:
            self._observed_level = level
            self._current_goal_level = level + 1
            return []
        if level <= self._observed_level:
            return []
        events: list[dict[str, Any]] = []
        start = int(self._start_level or 0)
        completed_grid = None
        try:
            from carnot.agentic.arc_agi3_world_model import grid_of
            from carnot.agentic.arc_executable_world_model import detect_cell, to_logical

            cell = detect_cell(grid_of(latest))
            completed_grid = to_logical(grid_of(latest), cell)
        except Exception:
            completed_grid = None
        for new_level in range(self._observed_level + 1, level + 1):
            relative = new_level - start
            if relative >= self.target_levels:
                continue
            event = self._begin_level_goal_episode(
                new_level,
                frames_seen=frames_seen,
                completed_grid=completed_grid,
            )
            events.append(event)
        self._observed_level = level
        return events

    def _begin_level_goal_episode(
        self, completed_level: int, *, frames_seen: int, completed_grid: Any = None
    ) -> dict[str, Any]:
        next_goal = int(completed_level) + 1
        self._current_goal_level = next_goal
        self._episode_transition_start = len(self.transitions)
        self._episode_dsl_transition_start = len(self._dsl_transitions)
        self.induced = False
        self.plan = []
        self.pi = 0
        self._level_reinduction_pending = True
        self._execute_plan_from_current = True
        self.world_model_trust_selection = None
        self.dsl_energy = None
        self.explorer.set_goal_bias(None, label="")
        if completed_grid is not None:
            try:
                import numpy as np

                self._previous_level_complete_grid = np.asarray(completed_grid).copy()
            except Exception:
                self._previous_level_complete_grid = None
        else:
            self._previous_level_complete_grid = None
        event = {
            "trigger": "level_up",
            "completed_level": int(completed_level),
            "next_goal_level": next_goal,
            "transition_start": int(self._episode_transition_start),
            "dsl_transition_start": int(self._episode_dsl_transition_start),
            "frames_seen": int(frames_seen),
            "win_state_exemplar_captured": self._previous_level_complete_grid is not None,
        }
        self.level_induction_events.append(event)
        return event

    def _current_goal_reached(self) -> bool:
        if self._current_goal_level is None:
            return self.explorer.best_level > (self.explorer.start_level or 0)
        return self.explorer.best_level >= self._current_goal_level

    def _should_enter_induction(self, *, stalled: bool, won: bool) -> tuple[bool, Optional[str]]:
        if (
            self._level_reinduction_pending
            and not self.induced
            and len(self.transitions) > self._episode_transition_start
        ):
            return True, "level_up_reinduction"
        if stalled and not won and not self.induced:
            return True, "stall"
        return False, None

    def _install_goal_bias(self, is_done) -> None:
        """Install the induced goal as a frontier bias for best-first ordering.

        GRADED-GOAL-ENERGY FIX (2026-06-25, opportunity 1): the induced goal predicate is a BINARY
        callable (grid -> bool). Installing it verbatim as ``1.0 if is_done else 0.0`` gives the
        best-first frontier a CLIFF -- zero signal at every non-terminal state, a spike only at the
        exact win -- so the search has no gradient to descend toward the goal until it accidentally
        lands on it. This degrades the live graded GoalSatisfactionEnergy and is consistent with the
        multi-level deepening null (lp85 L2 did not bank). When CARNOT_ARC_GRADED_GOAL_BIAS=1 and a
        win-state exemplar (the previous level's completion grid) is available, install a GRADED
        energy instead: E(grid) = 0.0 if is_done(grid) else the normalized cell-Hamming distance to
        the win exemplar -- a terminal anchor at the true goal PLUS a continuous descent gradient
        everywhere else, so the frontier can flow toward goal-shaped states. Default (env unset)
        keeps the binary behavior byte-identical (parity-safe).
        """
        if not callable(is_done):
            return

        import os

        import numpy as np

        exemplar = getattr(self, "_previous_level_complete_grid", None)
        if os.environ.get("CARNOT_ARC_GRADED_GOAL_BIAS") == "1" and exemplar is not None:
            exemplar_arr = np.asarray(exemplar)

            def _graded_bias(frame: Any) -> float:
                from carnot.agentic.arc_agi3_world_model import grid_of
                from carnot.agentic.arc_executable_world_model import to_logical

                grid = to_logical(grid_of(frame), self.cell)
                try:
                    if is_done(grid):
                        return 0.0
                except Exception:
                    pass
                g = np.asarray(grid)
                if g.shape != exemplar_arr.shape:
                    return 1.0  # mismatched shape -> maximally far
                return float(np.mean(g != exemplar_arr))

            label = f"L{self._current_goal_level or '?'}_induced_goal_graded_distance"
            self.explorer.set_goal_bias(_graded_bias, label=label, lower_is_better=True)
            return

        def _bias(frame: Any) -> float:
            from carnot.agentic.arc_agi3_world_model import grid_of
            from carnot.agentic.arc_executable_world_model import to_logical

            grid = to_logical(grid_of(frame), self.cell)
            return 1.0 if is_done(grid) else 0.0

        label = f"L{self._current_goal_level or '?'}_induced_goal_predicate"
        self.explorer.set_goal_bias(_bias, label=label)

    def _goal_energy_for_plan(self, is_done):
        """SCENARIO-ARC-WMTE-4821-LIVE-PLAN-WIRING: lambda-gated model planner energy."""

        if self.goal_guidance_lambda <= 0.0 or not callable(is_done):
            return None

        import os

        import numpy as np

        exemplar = getattr(self, "_previous_level_complete_grid", None)
        use_graded = os.environ.get("CARNOT_ARC_GRADED_GOAL_BIAS") == "1" and exemplar is not None
        exemplar_arr = np.asarray(exemplar) if use_graded else None
        scale = float(self.goal_guidance_lambda)

        def _energy(grid: Any) -> float:
            try:
                if is_done(grid):
                    return 0.0
            except Exception:
                pass
            if exemplar_arr is not None:
                g = np.asarray(grid)
                if g.shape != exemplar_arr.shape:
                    return scale
                return scale * float(np.mean(g != exemplar_arr))
            return scale

        return _energy

    @staticmethod
    def _planner_accepts_goal_energy(plan_in_model) -> bool:
        import inspect

        try:
            signature = inspect.signature(plan_in_model)
        except (TypeError, ValueError):
            return True
        if "goal_energy" in signature.parameters:
            return True
        return any(
            param.kind is inspect.Parameter.VAR_KEYWORD for param in signature.parameters.values()
        )

    def _call_plan_in_model(self, plan_in_model, engine, is_done, start_grid):
        goal_energy = self._goal_energy_for_plan(is_done)
        if goal_energy is None or not self._planner_accepts_goal_energy(plan_in_model):
            return plan_in_model(engine, is_done, start_grid)
        return plan_in_model(engine, is_done, start_grid, goal_energy=goal_energy)

    def _guided_plan_in_model(self, plan_in_model):
        def _wrapped(engine, is_done, start_grid):
            return self._call_plan_in_model(plan_in_model, engine, is_done, start_grid)

        return _wrapped

    def _next_plan_move(self) -> tuple:
        step = self.plan[self.pi]
        self.pi += 1
        return (step["action"], step.get("data"))

    def _remember_active_probe_origin(self, move: tuple, latest: Any) -> None:
        if self._active_probe_pending is None or latest is None:
            return
        kind, data = move
        if kind in ("RESET", None):
            return
        try:
            if int(kind) != int(self._active_probe_pending.action):
                return
            from carnot.agentic.arc_agi3_world_model import grid_of
            from carnot.agentic.arc_executable_world_model import detect_cell, to_logical

            self.cell = detect_cell(grid_of(latest))
            self._prev = (to_logical(grid_of(latest), self.cell), int(kind), data)
        except Exception:
            return

    def _observe_active_probe_transition(self, transition: Any) -> None:
        if self._active_probe_pending is None or self._active_probe_controller is None:
            return
        try:
            update = self._active_probe_controller.observe_transition(
                transition.grid,
                self._active_probe_pending,
                transition.next_grid,
            )
            self.active_probe_diagnostics = self._active_probe_controller.diagnostics()
            self.active_probe_diagnostics["last_update"] = {
                "posterior_entropy_before": round(float(update.posterior_entropy_before), 8),
                "posterior_entropy_after": round(float(update.posterior_entropy_after), 8),
                "posterior_entropy_reduction": round(
                    float(update.posterior_entropy_reduction),
                    8,
                ),
                "matched_hypotheses": list(update.matched_hypotheses),
            }
            self._pending_induction_reason = "active_probe_observed"
            self.induced = False
            self.phase = "induce"
        except Exception as exc:
            self.active_probe_diagnostics["last_update_error"] = repr(exc)[:160]
        finally:
            self._active_probe_pending = None

    def _proposer(self):
        if self.proposer is None:
            import os
            from carnot.agentic.arc_executable_world_model import LocalGGUFProposer

            # Live-submission generator (validated 2026-06-19): Qwen3.5-9B-MTP + MTP + 8-bit KV + /no_think,
            # n_predict>=2048. 5.9GB Q4 fits 16GB; 62.5% Layer-B grounding (DeepSeek-Flash 25%, gemma verbose).
            # Kaggle deploy: set CARNOT_ARC_GGUF_PATH to the bundled /kaggle/input/.../Qwen3.5-9B-Q4_K_M.gguf;
            # CARNOT_ARC_MTP=0 disables MTP if a tight-VRAM box needs the ~4GB the self-draft costs.
            # CARNOT_ARC_NGL (default 999=all layers on GPU): the operator prefill-to-RAM lever. Lowering it
            # keeps that many of the top weight layers in system RAM (mmap'd, prefilled in page cache) instead
            # of VRAM, freeing GPU memory for the q8 KV-cache + the live CNN dynamics fit that coexists with
            # the LLM on the shared 16GB eval GPU. Acceptable because the ARC eval has no internal time limit.
            self.proposer = LocalGGUFProposer(
                repo_substr="Qwen3.5-9B-MTP",
                model_path=os.environ.get("CARNOT_ARC_GGUF_PATH") or None,
                mtp=(os.environ.get("CARNOT_ARC_MTP", "1") != "0"),
                kv_quant="q8_0",
                no_think_prefix="/no_think\n",
                max_tokens=2560,
                n_gpu_layers=int(os.environ.get("CARNOT_ARC_NGL", "999")),
            )
        return self.proposer

    def _world_model_candidates(self, engine, is_done) -> list[WorldModelCandidate]:
        candidates = [WorldModelCandidate("loaded_world_model.py", engine, is_done)]
        # PoE-World (arXiv:2505.10819), OFF by default (CARNOT_ARC_POE_WORLD=1). Adds a weighted
        # product-of-experts engine as an extra candidate so select_trusted_world_model can rank it
        # against the single induced engine by held-out predictive fitness (oracle-distinct). Distinct
        # from the nulled max-vote ProductWorldModel (exp4749): weighted-consensus combination + fitted/
        # pruned weights. Wrapped in try/except so it can never break the live induction path.
        if os.environ.get("CARNOT_ARC_POE_WORLD") == "1":
            try:
                from carnot.agentic import arc_poe_world_model as poe

                active = self._active_transitions()
                if len(active) >= 4:
                    split = max(2, int(len(active) * 0.6))
                    model = poe.build_poe_world_model(
                        active[:split], active[split:] or active[:split]
                    )
                    if model.diagnostics_.get("n_kept", 0) > 0:
                        candidates.append(WorldModelCandidate("poe_world", model.engine, is_done))
            except Exception:
                pass
        provider = getattr(self.proposer, "world_model_candidates", None)
        if provider is None:
            provider = getattr(self.proposer, "candidate_engines", None)
        if not callable(provider):
            return candidates
        for i, row in enumerate(provider(self.short)):
            if isinstance(row, WorldModelCandidate):
                candidates.append(row)
            elif isinstance(row, dict):
                candidates.append(
                    WorldModelCandidate(
                        str(row.get("name") or f"candidate_{i}"),
                        row["engine"],
                        row.get("is_level_complete"),
                    )
                )
            else:
                name, candidate_engine, *rest = row
                candidates.append(
                    WorldModelCandidate(
                        str(name),
                        candidate_engine,
                        rest[0] if rest else None,
                    )
                )
        return candidates

    def next_move(self, frames, latest):
        from carnot.agentic.arc_executable_world_model import to_logical, detect_cell

        self._maybe_route_from_frame(latest)
        # collect a transition from the last action's outcome
        if self._prev is not None and latest is not None:
            from carnot.agentic.arc_agi3_world_model import grid_of
            from carnot.agentic.arc_executable_world_model import Transition

            g0, aid, data = self._prev
            g1 = to_logical(grid_of(latest), self.cell)
            transition = Transition(g0, aid, data, g1, 0, _level_of(latest))
            self.transitions.append(transition)
            self._dsl_transitions.append((g0, _action_key(aid, data), g1))
            self._observe_active_probe_transition(transition)
        boundary_events = self._observe_level_boundary(latest, frames_seen=len(frames))
        if boundary_events and latest is not None:
            try:
                from carnot.agentic.arc_agi3_world_model import grid_of

                self.cell = detect_cell(grid_of(latest))
                self.root_grid = to_logical(grid_of(latest), self.cell)
                self._prev = None
            except Exception:
                pass
        if self.phase == "explore":
            should_induce, reason = self._should_enter_induction(
                stalled=False,
                won=self._current_goal_reached(),
            )
            if should_induce:
                self.phase = "induce"
                self._pending_induction_reason = reason
                if reason != "level_up_reinduction":
                    self._execute_plan_from_current = False
            else:
                mv = self.explorer.next_move(frames, latest)
                if latest is not None:
                    from carnot.agentic.arc_agi3_world_model import grid_of

                    self.cell = detect_cell(grid_of(latest))
                    if self.root_grid is None and self.explorer.root is not None:
                        self.root_grid = to_logical(grid_of(latest), self.cell)
                    if mv[0] not in ("RESET", None):
                        self._prev = (to_logical(grid_of(latest), self.cell), int(mv[0]), mv[1])
                    else:
                        self._prev = None
                # VERIFIER-ROUTED CASCADE escalation: hand off to the tier-3 induction path on
                # a genuine stall, and also after a level-up once post-boundary evidence exists.
                won = self._current_goal_reached()
                stalled = len(self.transitions) >= self.explore_budget or self.explorer.explored_out
                should_induce, reason = self._should_enter_induction(stalled=stalled, won=won)
                if should_induce:
                    self.phase = "induce"
                    self._pending_induction_reason = reason
                    if reason != "level_up_reinduction":
                        self._execute_plan_from_current = False
                return mv
        if self.phase == "induce" and not self.induced:
            self.induced = True
            self._induce_and_plan()
            self._level_reinduction_pending = False
            self.phase = "execute" if self.plan else "explore"
            self._prev = None
            if self.plan:
                if self._execute_plan_from_current:
                    mv = self._next_plan_move()
                    self._remember_active_probe_origin(mv, latest)
                    return mv
                return ("RESET", None)
            return self.explorer.next_move(frames, latest)
        if self.phase == "execute" and self.pi < len(self.plan):
            mv = self._next_plan_move()
            self._remember_active_probe_origin(mv, latest)
            return mv
        # plan exhausted / no model -> keep exploring
        self.phase = "explore"
        return self.explorer.next_move(frames, latest)

    def _induce_and_plan(self):
        import os

        from carnot.agentic import arc_executable_world_model as e3

        active_transitions = self._active_transitions()
        attempt = {
            "reason": self._pending_induction_reason or "stall",
            "goal_level": self._current_goal_level,
            "transition_count": len(active_transitions),
            "dsl_transition_count": len(self._active_dsl_transitions()),
            "model_specs": "offline_dsl_induction_no_llm",
            "planned": False,
            "skipped": "",
        }
        self.induction_attempts.append(attempt)

        # Production-safe escape hatch (default OFF): when CARNOT_ARC_DISABLE_INDUCTION=1, skip the LLM
        # world-model induction tier entirely and stay in the (fast) tier-1 explorer. The local submission
        # GATE sets this so it measures the explorer's SEARCH/efficiency cleanly + fast, without paying the
        # ~30s+ llama-server spawn (a one-time, acceptable cost under the real 12h eval, but it dominates a
        # bounded local gate run and is irrelevant to detecting a SEARCH regression). Unset in production
        # (Kaggle) -> induction runs exactly as before.
        if os.environ.get("CARNOT_ARC_DISABLE_INDUCTION") == "1":
            attempt["skipped"] = "disabled_by_env"
            return
        if not active_transitions:
            attempt["skipped"] = "no_active_transitions"
            return

        try:
            if self.program_synthesis_filter_enabled:
                try:
                    filter_result = induce_action_effect_proposal_filter(
                        game=self.short,
                        transitions=active_transitions,
                        proposer=self._proposer(),
                        cell=self.cell,
                        trust_threshold=self.program_synthesis_filter_trust_threshold,
                    )
                    self.explorer.set_program_synthesis_filter(filter_result.proposal_filter)
                    attempt["program_synthesis_filter_used"] = True
                    attempt["program_synthesis_filter_residual"] = filter_result.residual
                    attempt["heldout_programs_kept"] = int(filter_result.heldout_programs_kept)
                    attempt["heldout_programs_rejected"] = int(
                        filter_result.heldout_programs_rejected
                    )
                    attempt["program_trust_weights"] = list(filter_result.program_trust_weights)
                    attempt["program_synthesis_filter_diagnostics"] = (
                        self.explorer.program_synthesis_filter_diagnostics()
                    )
                except Exception as filter_exc:
                    attempt["program_synthesis_filter_used"] = False
                    attempt["program_synthesis_filter_error"] = repr(filter_exc)[:160]
            # PRIOR-WARM-STARTED LEARNED ENGINE (2026-06-21): try the per-game world model LEARNED from the
            # played transitions (warm-started from the cross-game CNN prior that transfers 5/5 to unseen
            # games), GATED by the same >=0.5 held-out trust bar the LLM path uses. If it earns trust and
            # plans a win, use it (zero-LLM, execution-grounded); otherwise fall through to the LLM
            # induction. Non-fatal -- never breaks the existing path. The prior is models/arc_dynamics_prior.pt.
            try:
                from carnot.agentic.arc_live_ttt import gated_engine_from_transitions

                _eng, _isdone, _diag = gated_engine_from_transitions(self.short, active_transitions)
                attempt["ttt_prior_engine"] = _diag
                if _eng is not None and self.root_grid is not None:
                    self._install_goal_bias(_isdone)
                    _plan = self._call_plan_in_model(
                        e3.plan_in_model,
                        _eng,
                        _isdone,
                        self.root_grid,
                    )
                    if _plan:
                        self.plan = _plan
                        attempt["planned"] = True
                        attempt["plan_length"] = len(_plan)
                        attempt["engine_source"] = "ttt_prior_warmstarted"
                        return
            except Exception as _ttt_e:
                attempt["ttt_prior_engine_error"] = repr(_ttt_e)[:120]
            next_level_episode = (
                self._previous_level_complete_grid is not None
                and self._current_goal_level is not None
                and self._start_level is not None
                and int(self._current_goal_level) > int(self._start_level) + 1
            )
            if attempt["reason"] == "level_up_reinduction" or next_level_episode:
                attempt["reason"] = "level_up_reinduction"
                self._fit_dsl_model()
                structural_goal_provider = None
                try:
                    from carnot.agentic.arc_value_learner import structural_alignment_goal_candidate

                    structural_goal_provider = structural_alignment_goal_candidate
                except Exception:
                    structural_goal_provider = None
                reinduction_proposer = self._proposer()
                reinduction_load_engine = e3.load_engine
                attempt["structured_engine_enabled"] = False
                if os.environ.get("CARNOT_ARC_STRUCTURED_ENGINE") == "1":
                    from carnot.agentic import arc_structured_world_model as structured_wm

                    reinduction_load_engine = structured_wm.make_structured_load_engine(
                        game=self.short,
                        transitions=active_transitions,
                        proposer=reinduction_proposer,
                        cell=self.cell,
                        trust_threshold=self.factored_trust_threshold,
                        fallback_goal_loader=e3.load_engine,
                    )
                    reinduction_proposer = structured_wm.StructuredEngineReinductionProposer(
                        reinduction_proposer
                    )
                    attempt["structured_engine_enabled"] = True
                outcome = execute_bounded_llm_reinduction(
                    game=self.short,
                    transitions=active_transitions,
                    cell=self.cell,
                    root_grid=self.root_grid,
                    proposer=reinduction_proposer,
                    candidate_provider=self._world_model_candidates,
                    load_engine=reinduction_load_engine,
                    plan_in_model=self._guided_plan_in_model(e3.plan_in_model),
                    max_rounds=MAX_REFINEMENT_ROUNDS,
                    min_heldout_accuracy=1.0,
                    previous_level_complete_grid=self._previous_level_complete_grid,
                    enable_subgoal_search=self.subgoal_search,
                    subgoal_budget=self.subgoal_budget,
                    value_head=self.value_head,
                    enable_factored_planner=self.factored_planner,
                    factored_trust_threshold=self.factored_trust_threshold,
                    structural_goal_provider=structural_goal_provider,
                )
                attempt.update(
                    {
                        "model_specs": outcome.model_specs,
                        "planned": bool(outcome.planned),
                        "skipped": outcome.skipped,
                        "plan_length": len(outcome.plan),
                        "selected_candidate_name": outcome.selected_candidate_name,
                        "goal_candidate_names": list(outcome.goal_candidate_names),
                        "dynamics_candidate_names": list(outcome.dynamics_candidate_names),
                        "refinement_rounds_used": int(outcome.refinement_rounds_used),
                        "refinement_rounds": list(outcome.rounds),
                        "counterexamples": list(outcome.counterexamples),
                        "verifier_is_oracle": bool(outcome.verifier_is_oracle),
                        "win_state_exemplar_injected": self._previous_level_complete_grid
                        is not None,
                        "goal_predicate_satisfiable": bool(outcome.goal_predicate_satisfiable),
                        "goal_satisfiability": dict(outcome.goal_satisfiability),
                        "goal_expression": outcome.goal_expression,
                        "structural_goal_diagnostics": dict(outcome.structural_goal_diagnostics),
                        "subgoal_search_used": bool(
                            outcome.subgoal_search_used or outcome.subgoal_decomposition
                        ),
                        "subgoal_decomposition": list(outcome.subgoal_decomposition),
                        "per_subgoal_reachable": list(outcome.per_subgoal_reachable),
                        "factored_planner_used": bool(outcome.factored_planner_used),
                        "expert_trust_weights": list(outcome.expert_trust_weights),
                    }
                )
                if outcome.goal_predicate is not None:
                    self._install_goal_bias(outcome.goal_predicate)
                if outcome.planned:
                    self.plan = list(outcome.plan)
                return
            self._fit_dsl_model()
            if self.active_probe_controller_enabled and self.root_grid is not None:
                try:
                    from carnot.agentic.arc_active_probe import (
                        ActiveProbeController,
                        augment_with_transition_baselines,
                        make_hypothesis_posterior,
                        probe_actions_from_model_candidates,
                    )

                    candidate_pool = augment_with_transition_baselines([], active_transitions)
                    if self._active_probe_controller is None:
                        self._active_probe_controller = ActiveProbeController(
                            make_hypothesis_posterior(candidate_pool),
                            probe_budget=self.active_probe_budget,
                            concentration_threshold=self.active_probe_concentration_threshold,
                        )
                    controller = self._active_probe_controller
                    actions = probe_actions_from_model_candidates(
                        e3._model_candidates(self.root_grid)
                    )
                    chosen_probe = controller.choose_probe(self.root_grid, actions)
                    attempt["active_probe_enabled"] = True
                    attempt["active_probe_candidate_names"] = [
                        str(candidate.name) for candidate in candidate_pool
                    ]
                    attempt["active_probe_diagnostics"] = controller.diagnostics()
                    if chosen_probe is not None and chosen_probe.expected_information_gain > 0.0:
                        self.plan = [chosen_probe.action.as_plan_step()]
                        self.pi = 0
                        self._active_probe_pending = chosen_probe.action
                        self._execute_plan_from_current = True
                        attempt["planned"] = True
                        attempt["plan_length"] = 1
                        attempt["engine_source"] = "active_probe_pre_llm_disambiguation"
                        attempt["active_probe_action"] = chosen_probe.action.as_plan_step()
                        attempt["active_probe_expected_information_gain"] = round(
                            float(chosen_probe.expected_information_gain),
                            8,
                        )
                        attempt["active_probe_energy_score"] = round(
                            float(chosen_probe.energy_score),
                            8,
                        )
                        attempt["active_probe_prediction_buckets"] = list(
                            chosen_probe.prediction_buckets
                        )
                        self.active_probe_diagnostics = controller.diagnostics()
                        return
                    self.active_probe_diagnostics = controller.diagnostics()
                except Exception as probe_exc:
                    attempt["active_probe_error"] = repr(probe_exc)[:160]
            ok, _ = self._proposer().induce(self.short, active_transitions, self.cell)
            if not ok or self.root_grid is None:
                attempt["skipped"] = "proposer_failed_or_missing_root"
                return
            engine, is_done = e3.load_engine(self.short)
            candidate_pool = self._world_model_candidates(engine, is_done)
            if self.active_probe_controller_enabled:
                try:
                    from carnot.agentic.arc_active_probe import (
                        ActiveProbeController,
                        augment_with_transition_baselines,
                        make_hypothesis_posterior,
                        probe_actions_from_model_candidates,
                    )

                    candidate_pool = augment_with_transition_baselines(
                        candidate_pool,
                        active_transitions,
                    )
                    if self._active_probe_controller is None:
                        self._active_probe_controller = ActiveProbeController(
                            make_hypothesis_posterior(candidate_pool),
                            probe_budget=self.active_probe_budget,
                            concentration_threshold=self.active_probe_concentration_threshold,
                        )
                    controller = self._active_probe_controller
                    actions = probe_actions_from_model_candidates(
                        e3._model_candidates(self.root_grid)
                    )
                    chosen_probe = controller.choose_probe(self.root_grid, actions)
                    attempt["active_probe_enabled"] = True
                    attempt["active_probe_candidate_names"] = [
                        str(candidate.name) for candidate in candidate_pool
                    ]
                    attempt["active_probe_diagnostics"] = controller.diagnostics()
                    if chosen_probe is not None and chosen_probe.expected_information_gain > 0.0:
                        self.plan = [chosen_probe.action.as_plan_step()]
                        self.pi = 0
                        self._active_probe_pending = chosen_probe.action
                        self._execute_plan_from_current = True
                        attempt["planned"] = True
                        attempt["plan_length"] = 1
                        attempt["engine_source"] = "active_probe_disambiguation"
                        attempt["active_probe_action"] = chosen_probe.action.as_plan_step()
                        attempt["active_probe_expected_information_gain"] = round(
                            float(chosen_probe.expected_information_gain),
                            8,
                        )
                        attempt["active_probe_energy_score"] = round(
                            float(chosen_probe.energy_score),
                            8,
                        )
                        attempt["active_probe_prediction_buckets"] = list(
                            chosen_probe.prediction_buckets
                        )
                        self.active_probe_diagnostics = controller.diagnostics()
                        return
                    best = controller.posterior.best_candidate()
                    if best is not None:
                        engine = best.engine
                        is_done = best.is_level_complete or is_done
                        attempt["active_probe_committed_hypothesis"] = str(best.name)
                        attempt["active_probe_diagnostics"] = controller.diagnostics()
                        self.active_probe_diagnostics = controller.diagnostics()
                except Exception as probe_exc:
                    attempt["active_probe_error"] = repr(probe_exc)[:160]
            if self.short in HIDDEN_STATE_GAME_IDS:
                self.world_model_trust_selection = select_trusted_world_model(
                    active_transitions,
                    candidate_pool,
                    hidden_state=True,
                )
                trust_score = self.world_model_trust_selection.selected_score
                attempt["trust_energy"] = round(float(trust_score.trust_energy), 6)
                attempt["heldout_change_consistency"] = round(
                    float(trust_score.heldout_change_consistency), 6
                )
                attempt["heldout_accuracy"] = round(float(trust_score.heldout_accuracy), 6)
                attempt["correct_changed_cells"] = int(trust_score.correct_changed_cells)
                attempt["binary_gate_pass"] = bool(trust_score.binary_gate_pass)
                if not trust_score.trust_pass:
                    attempt["skipped"] = "hidden_state_trust_below_threshold"
                    return
                engine = self.world_model_trust_selection.selected.engine
                is_done = self.world_model_trust_selection.selected.is_level_complete or is_done
            else:
                import os

                vr = e3.WorldModelVerifier(active_transitions).score(engine)
                # CARNOT_ARC_TRUST_METRIC=cell_recall gates on GRADED changed-cell recall instead of the
                # exact-FULL-GRID match (the coordinated-redesign lever for the 0.08 wall: exact-match reads
                # ~0 for an imperfect-but-useful induced model and gates it out -> the induce->plan path is a
                # no-op). Default 'exact' preserves the submitted behavior + the parity test. Both metrics are
                # recorded on the attempt for diagnosis regardless of which one gates.
                _metric = os.environ.get("CARNOT_ARC_TRUST_METRIC", "exact")
                _gate_value = vr.cell_recall if _metric == "cell_recall" else vr.accuracy
                attempt["verify_accuracy"] = round(vr.accuracy, 4)
                attempt["verify_cell_recall"] = round(vr.cell_recall, 4)
                attempt["trust_metric"] = _metric
                if _gate_value < 0.5:  # too weak to trust for execution-grounded planning
                    attempt["skipped"] = "world_model_accuracy_below_threshold"
                    return
            self._install_goal_bias(is_done)
            # plan ENTIRELY in the model (zero real actions); execute phase RESETs then
            # replays this plan in the real env, halting on divergence.
            plan = self._call_plan_in_model(e3.plan_in_model, engine, is_done, self.root_grid)
            if plan:
                self.plan = plan
                attempt["planned"] = True
                attempt["plan_length"] = len(plan)
                return
            if self.subgoal_search:
                subgoals = propose_hierarchical_subgoals(
                    game=self.short,
                    transitions=active_transitions,
                    proposer=self._proposer(),
                    previous_level_complete_grid=self._previous_level_complete_grid,
                    max_subgoals=self.subgoal_budget,
                )
                subgoal_result = plan_hierarchical_subgoals(
                    engine=engine,
                    final_goal=is_done,
                    start_grid=self.root_grid,
                    subgoals=subgoals,
                    plan_in_model=self._guided_plan_in_model(e3.plan_in_model),
                    value_head=self.value_head,
                    max_subgoals=self.subgoal_budget,
                )
                attempt["subgoal_search_used"] = True
                attempt["subgoal_decomposition"] = list(subgoal_result.subgoal_decomposition)
                attempt["per_subgoal_reachable"] = list(subgoal_result.per_subgoal_reachable)
                attempt["subgoal_residual"] = subgoal_result.residual
                attempt["hierarchical_plan_length"] = len(subgoal_result.plan)
                if subgoal_result.planned:
                    self.plan = list(subgoal_result.plan)
                    attempt["planned"] = True
                    attempt["plan_length"] = len(self.plan)
            if self.factored_planner:
                expert_result = e3.induce_programmatic_object_experts(
                    game=self.short,
                    transitions=active_transitions,
                    proposer=self._proposer(),
                    cell=self.cell,
                    trust_threshold=self.factored_trust_threshold,
                )
                subgoals = propose_hierarchical_subgoals(
                    game=self.short,
                    transitions=active_transitions,
                    proposer=self._proposer(),
                    previous_level_complete_grid=self._previous_level_complete_grid,
                    max_subgoals=self.subgoal_budget,
                )
                factored_result = e3.plan_factored_subgoal_sequence(
                    start_grid=self.root_grid,
                    final_goal=is_done,
                    experts=expert_result.experts,
                    subgoals=subgoals,
                    value_head=self.value_head,
                    max_subgoals=self.subgoal_budget,
                )
                attempt["factored_planner_used"] = True
                attempt["expert_trust_weights"] = list(expert_result.expert_trust_weights)
                attempt["factored_subgoal_decomposition"] = list(
                    factored_result.subgoal_decomposition
                )
                attempt["factored_per_subgoal_reachable"] = list(
                    factored_result.per_subgoal_reachable
                )
                attempt["factored_residual"] = factored_result.residual or expert_result.residual
                if factored_result.planned:
                    self.plan = list(factored_result.plan)
                    attempt["planned"] = True
                    attempt["plan_length"] = len(self.plan)
        except Exception:
            attempt["skipped"] = "exception"
            return

    def is_done(self, frames, latest):
        return self.explorer.is_done(frames, latest) and self.phase == "explore"


# ============================================================================================
# SINGLE SOURCE OF TRUTH for WHAT SHIPS. The 0.08 incident (2026-06-19): the offline eval measured
# STRONGER opt-in configs (explorer_bf unlocked cn04) while the SUBMITTED default shipped bare BFS,
# and nobody caught it because "better" was opt-in-only and the headline metric was banked-replay
# levels, not the submitted path. RULE (enforced by test_arc_submitted_agent_parity.py): the
# STRONGEST measured config MUST be this declared submitted config -- improvements go HERE, never to
# an opt-in-only eval flag. The eval's submission-baseline + the parity test both read this dict, so
# the shipped agent and the measured baseline can never silently diverge again.
SUBMITTED_AGENT_CONFIG = {
    "policy": "E3AgentPolicy",  # the verifier-routed cascade (NOT cascade=False banked-replay)
    "cascade": True,
    # explorer config the live agent actually runs.
    "value_weight": SUBMITTED_VALUE_WEIGHT,
    "target_levels": SUBMITTED_TARGET_LEVELS,
    "search_mode": SUBMITTED_SEARCH_MODE,
    "graph_explore_budget": SUBMITTED_GRAPH_EXPLORE_BUDGET,
    "routed_explore_budget": SUBMITTED_ROUTED_EXPLORE_BUDGET,
    "lazy_value_top_k": SUBMITTED_LAZY_VALUE_TOP_K,
    "frontier_batch_size": SUBMITTED_FRONTIER_BATCH_SIZE,
    "navigation_cost_tiebreak": SUBMITTED_NAVIGATION_COST_TIEBREAK,
    "frame_change_predictor_enabled": SUBMITTED_FRAME_CHANGE_PREDICTOR_ENABLED,
    "frame_change_ranking_mode": SUBMITTED_FRAME_CHANGE_RANKING_MODE,
    "frame_change_prune_threshold": None,
    "action_effect_expansion_prior_enabled": SUBMITTED_ACTION_EFFECT_EXPANSION_PRIOR_ENABLED,
    "action_effect_expansion_prior_mode": SUBMITTED_ACTION_EFFECT_EXPANSION_PRIOR_MODE,
    "goal_energy_enabled": SUBMITTED_GOAL_ENERGY_ENABLED,
    "goal_energy_wired": True,
    "goal_energy_source": GOAL_ENERGY_SOURCE,
    "goal_energy_alpha": SUBMITTED_GOAL_ENERGY_ALPHA,
    "goal_energy_beta": SUBMITTED_GOAL_ENERGY_BETA,
    "goal_guidance_lambda": SUBMITTED_GOAL_GUIDANCE_LAMBDA,
    "goal_energy_candidate_guidance_enabled": (SUBMITTED_GOAL_ENERGY_CANDIDATE_GUIDANCE_ENABLED),
    "goal_energy_candidate_guidance_alpha": (SUBMITTED_GOAL_ENERGY_CANDIDATE_GUIDANCE_ALPHA),
    "goal_energy_candidate_guidance_beta": SUBMITTED_GOAL_ENERGY_CANDIDATE_GUIDANCE_BETA,
    "qd_generation_enabled": SUBMITTED_QD_GENERATION_ENABLED,
    "qd_generation_mode": SUBMITTED_QD_GENERATION_MODE,
    "controllable_novelty_proposal_enabled": SUBMITTED_CONTROLLABLE_NOVELTY_PROPOSAL_ENABLED,
    "controllable_novelty_proposal_mode": SUBMITTED_CONTROLLABLE_NOVELTY_MODE,
    "object_centric_proposal_enabled": SUBMITTED_OBJECT_CENTRIC_PROPOSAL_ENABLED,
    "object_centric_proposal_mode": SUBMITTED_OBJECT_CENTRIC_PROPOSAL_MODE,
    "color_blob_salience_enabled": SUBMITTED_COLOR_BLOB_SALIENCE_ENABLED,
    "color_blob_salience_mode": SUBMITTED_COLOR_BLOB_SALIENCE_MODE,
    "program_synthesis_proposal_filter_enabled": (
        SUBMITTED_PROGRAM_SYNTHESIS_PROPOSAL_FILTER_ENABLED
    ),
    "program_synthesis_proposal_filter_trust_threshold": (
        SUBMITTED_PROGRAM_SYNTHESIS_PROPOSAL_FILTER_TRUST_THRESHOLD
    ),
    "amortized_first_contact_prior_enabled": SUBMITTED_AMORTIZED_FIRST_CONTACT_PRIOR_ENABLED,
    "amortized_first_contact_prior_mode": SUBMITTED_AMORTIZED_FIRST_CONTACT_PRIOR_MODE,
    "go_explore_archive_enabled": SUBMITTED_GO_EXPLORE_ARCHIVE_ENABLED,
    "go_explore_archive_mode": SUBMITTED_GO_EXPLORE_ARCHIVE_MODE,
    "ige_cell_selection_enabled": SUBMITTED_IGE_CELL_SELECTION_ENABLED,
    "ige_cell_selection_mode": SUBMITTED_IGE_CELL_SELECTION_MODE,
    "matm_similarity_retrieval_enabled": SUBMITTED_MATM_SIMILARITY_RETRIEVAL_ENABLED,
    "matm_similarity_retrieval_mode": SUBMITTED_MATM_SIMILARITY_RETRIEVAL_MODE,
    "router_wired": True,
    "solve_learning_router_wired": True,
    "strategy_router_enabled": True,
    "discriminative_router_wired": True,
    "discriminative_candidate_router_enabled": True,
    "candidate_router": "cross_game_discriminative_v3_tiebreaker",
    "verifier_is_oracle": False,
    "value_head_feature_subset": SUBMITTED_VALUE_HEAD_FEATURE_SUBSET,
    "value_head_checkpoint": DAGGER_VALUE_HEAD_RELATIVE_PATH,
    "value_head_distribution_corrected": True,
    "hierarchical_subgoal_search_enabled": False,
    "hierarchical_subgoal_budget": 3,
    "factored_planner_enabled": False,
    "factored_trust_threshold": 0.75,
    "world_model_dsl_wired": True,
    "online_discriminative": True,
    "dense_curiosity_progress_loop_enabled": False,
    "dense_curiosity_weight": 0.15,
    "dense_curiosity_discount": 0.5,
    "live_submit_package_path": "results/experiment_4643_submission_package_operator_resubmit.json",
    "live_submit_source": "experiment_4643_refresh_submission_package",
    "frozen_generator": {
        "model_id": "unsloth/Qwen3.5-9B-MTP-GGUF",
        "repo_substr": "Qwen3.5-9B-MTP",
        "model_filename": "Qwen3.5-9B-Q4_K_M.gguf",
        "model_path_env": "CARNOT_ARC_GGUF_PATH",
        "server_path_env": "CARNOT_LLAMA_SERVER",
        "llama_server_kind": "cuda-12.8-binary",
        "binary_not_wheel": True,
        "required_shared_libraries": [
            "libllama-common",
            "libllama",
            "libggml",
            "libggml-cuda",
        ],
        "mtp": True,
        "spec_type": "draft-mtp",
        "kv_quant": "q8_0",
        "no_think_prefix": "/no_think\n",
        "max_tokens": 2560,
        "n_predict_min": 2048,
        "port_strategy": "free_non_8919",
        "props_verify_endpoint": "/props",
        "wheel_fallback_allowed": False,
        "forbidden_models": ["gemma-8919"],
        "forbidden_gpu_targets": ["3090"],
        "gpu_target": "kaggle_cuda_gpu_not_3090",
    },
    "feature_router_enabled": False,
    "explore_diversity_default": False,
    "bare_control_config": {
        "policy": "E3AgentPolicy",
        "target_levels": 1,
        "value_weight": 0.0,
        "search_mode": SUBMITTED_SEARCH_MODE,
        "candidate_router": None,
        "navigation_cost_tiebreak": False,
        "action_effect_expansion_prior_enabled": False,
        "goal_energy_enabled": False,
        "goal_energy_candidate_guidance_enabled": False,
    },
}


def make_carnot_agent(base_cls, cascade: bool = True, proposer=None):
    """Adapt the Carnot policy onto the real ARC-AGI-3-Agents `Agent` base class.
    Submission: `from agents.agent import Agent; CarnotAgent = make_carnot_agent(Agent)`.

    cascade=True (DEFAULT, the competition path): the VERIFIER-ROUTED CASCADE
    (E3AgentPolicy) — tier-1 training-free explorer; on STALL escalate to tier-3 E3
    induction with the bundled open proposer (the verifier routes + grounds). This is the
    unified choose_action the hard eval needs. cascade=False: pure recognize-and-replay
    (dev/known games only — useless on the hidden eval).

    The exact shipped config is declared in SUBMITTED_AGENT_CONFIG (single source of truth);
    test_arc_submitted_agent_parity.py asserts this function's default policy matches it, so a
    silent divergence between what we measure and what we ship cannot recur (the 0.08 incident)."""

    class CarnotAgent(base_cls):  # type: ignore
        # The framework's Agent.MAX_ACTIONS default is 80 ("avoid looping forever"),
        # which is far too low for the eval: our deepest banked replay (lp85 -> L5)
        # alone needs well over 80 actions, and the held-out-game explore fallback
        # needs room to probe + solve several levels. 80 would truncate even our best
        # known game. The real bound is the eval's wall-clock budget (<=12h across all
        # games), not this per-game loop guard; Playback overrides it to 1e6 for the
        # same reason, so it is an intended override point. 400 comfortably covers our
        # multi-level replays + explore while staying well inside the time budget.
        MAX_ACTIONS = 400

        def __init__(self, *args, **kwargs) -> None:
            super().__init__(*args, **kwargs)
            gid = getattr(self, "game_id", "")
            self._policy = (
                E3AgentPolicy(
                    gid,
                    proposer=proposer,
                    target_levels=int(SUBMITTED_AGENT_CONFIG["target_levels"]),
                    value_weight=float(SUBMITTED_AGENT_CONFIG["value_weight"]),
                    search_mode=str(SUBMITTED_AGENT_CONFIG["search_mode"]),
                    lazy_value_top_k=int(SUBMITTED_AGENT_CONFIG["lazy_value_top_k"]),
                    frontier_batch_size=SUBMITTED_AGENT_CONFIG["frontier_batch_size"],
                    navigation_cost_tiebreak=bool(
                        SUBMITTED_AGENT_CONFIG["navigation_cost_tiebreak"]
                    ),
                    similarity_retrieval=bool(
                        SUBMITTED_AGENT_CONFIG["matm_similarity_retrieval_enabled"]
                    ),
                )
                if cascade
                else CarnotAgentPolicy(gid, load_solutions())
            )

        def is_done(self, frames, latest_frame) -> bool:
            return self._policy.is_done(frames, latest_frame)

        def choose_action(self, frames, latest_frame):
            from arcengine import GameAction

            kind, data = self._policy.next_move(frames, latest_frame)
            if kind == "RESET" or kind is None:
                return GameAction.RESET
            act = getattr(GameAction, f"ACTION{kind}")
            if data:
                # CRITICAL: GameAction.set_data() RETURNS the inner ComplexAction, NOT
                # the enum member. The framework's do_action_request reads
                # `action.action_data` off the object choose_action returns, so we must
                # mutate the enum in place and return the ENUM. Returning set_data()'s
                # result (a ComplexAction, which has no .action_data) crashes every
                # coordinate/click action against the real harness. game_id is a required
                # ComplexAction field (Playback injects it too), so carry it through.
                act.set_data({"game_id": getattr(self, "game_id", ""), **data})
            return act

    return CarnotAgent
