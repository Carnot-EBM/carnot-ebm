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
from carnot.agentic.arc_color_blob_salience import ColorBlobSaliencePrior, connected_color_blobs
from carnot.agentic.arc_dense_curiosity_progress import DenseCuriosityProgress
from carnot.agentic.arc_hud_bar_detector import (
    HUD_MASK_GUARD_MAX_SPLIT_NODES,
    MaskCollapseGuard,
    mask_summary,
)
from carnot.agentic.arc_hud_bar_detector import (
    # Aliased on import: `StepwiseExplorer.__init__` takes a BOOL kwarg spelled
    # `edge_bar_hud_mask`, and an unaliased import of the same name would be shadowed inside
    # that method -- confusing to read even though module-scope callers are unaffected.
    edge_bar_hud_mask as compute_edge_bar_hud_mask,
)
from carnot.agentic.arc_energy_fitness_qd import coerce_qd_generator
from carnot.agentic.arc_controllable_novelty import coerce_controllable_novelty_policy
from carnot.agentic.arc_go_explore import coerce_go_explore_archive
from carnot.agentic.arc_ige_cell_selector import coerce_ige_cell_selector
from carnot.agentic.arc_program_synthesis_filter import (
    coerce_program_synthesis_filter,
    induce_action_effect_proposal_filter,
)
from carnot.agentic.arc_inert_click_pruner import coerce_inert_click_pruner
from carnot.agentic.arc_object_history_salience import coerce_object_history_salience_prior
from carnot.agentic.arc_epistemic_ledger import coerce_epistemic_ledger
from carnot.agentic.arc_component_sampling import (
    component_partition as click_component_partition,
    redraw_component_pixel as redraw_click_component_pixel,
)
from carnot.agentic.arc_frontier_discipline import (
    TIER_COUNT as FRONTIER_TIER_COUNT,
    TierExhaustionPolicy,
    annotate_tiers as annotate_frontier_tiers,
    frontier_distance_field,
    nearest_open_node,
)
from carnot.agentic.arc_structured_evidence_memory import coerce_structured_evidence_memory
from carnot.agentic.arc_generic_causal_primitives import coerce_generic_causal_primitive
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
    _goal_satisfiability_check,
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
SUBMITTED_COLOR_BLOB_SALIENCE_ENABLED = False  # 2026-07-16: re-validated -- see below
SUBMITTED_COLOR_BLOB_SALIENCE_MODE = "single_color_connected_component_tiers"
# REQ-ARC-FCP-5699-11 (operator: "wire SGE into the live path", 2026-07-15): the LLM
# Strategy-Guided Exploration router (arXiv:2603.02045, arc_llm_strategy_proposer.py) is
# now genuinely REACHABLE from this live entrypoint via _load_submitted_candidate_router(),
# reusing the frozen live-submission generator (Qwen3.5-9B-MTP) through LocalGGUFProposer's
# existing port-based server-reuse -- no second model load, no extra Kaggle VRAM. Default
# False: the offline REQ-ARC-FCP-5699-3..10 investigation found SGE's anti-stagnation
# mechanism works correctly (nudge fires, taboo filtering works) but never demonstrated a
# capability WIN over the deterministic discriminative router in its own diagnostic harness
# -- this flag makes it a genuine, testable, opt-in alternative (matching the
# SUBMITTED_COLOR_BLOB_SALIENCE_ENABLED pattern), not the active default. Re-enable only
# after a real matched-budget A/B on the ACTUAL live path shows a win.
SUBMITTED_SGE_CANDIDATE_ROUTER_ENABLED = False
# Disabled pending re-validation (2026-07-14, submission-prep pre-flight): this flag caused a
# severe near-hang on the local submission gate (7/8 canonical games timed out at 115s/game,
# 0 solved vs baseline's 4) -- root cause was O(candidates x grid_cells) per-frame recomputation
# in ColorBlobSaliencePrior.score()/action_tier_rows() (fixed in arc_color_blob_salience.py, see
# that fix's own comments), but even after the fix this feature is still meaningfully slower per
# step than the baseline (measured: lp85 budget=500 took 68s, vs baseline's ~7761 actions/115s --
# roughly 9x slower per action) for ZERO measured benefit (three follow-on live-path level-up
# attempts using it, this same day, all returned honest_null). Re-enable only after a fresh
# matched-budget A/B shows a real win that justifies the residual per-step cost.
#
# RE-VALIDATED 2026-07-16 (REQ-ARC-FCP-5699 item 2) -- performance is FIXED, stays disabled
# for a DIFFERENT, more precise reason. Profiling found a second, separate O(candidates x
# grid_cells) recomputation the 2026-07-14 fix missed: arc_frame_change_predictor.rank_arc_
# actions -> _prior_value calls ColorBlobSaliencePrior.score(frame, candidate) via the generic
# two-arg protocol (no blobs/color_counts cache args), which alone accounted for 8176 calls to
# connected_color_blobs() on a single 500-action lp85 episode (23.1s of a 43.3s total run).
# Fixed with two changes in arc_color_blob_salience.py: (1) connected_color_blobs() vectorized
# via scipy.ndimage.label (200+300 randomized-grid equivalence tests confirm exact field-for-
# field agreement with the original pure-Python flood-fill), and (2) a bounded module-level
# per-frame cache so the generic two-arg score() callers stop recomputing per-candidate. Net:
# lp85 budget=500 went 68s -> 8.4s (baseline is 6.1s) -- the wall-clock/timeout problem is
# genuinely solved (the full local submission gate now runs all 8 games with ZERO timeouts,
# where it previously timed out on 7/8). BUT the gate's real verdict flipped from
# "TIMED OUT" to a clean "FAIL: REGRESSION: lost CORE solves ['m0r0']" -- with the confounding
# timeout removed, a genuine BEHAVIORAL regression is now visible for the first time: the
# salience-reordered exploration causes the agent to lose a game it otherwise reliably solves.
# This stays disabled because of THAT regression, not because it is too slow -- the speed fix
# is real and kept (it makes any future re-validation of this feature cheap to re-run), but
# re-enabling now requires investigating/fixing the m0r0 regression specifically, not another
# performance pass.
# Task 10 follow-on (2026-07-13): ObjectHistorySaliencePrior wraps action_prior with a
# per-object_hash change-history bonus (REQ-ARC-FCP-5591-2) but OFF by default, matching every
# other freshly-wired-but-unvalidated component in this file. Flipping this for the SCORED agent
# needs its own matched-budget offline A/B first, per the solve_rate_dropped guardrail. See
# python/carnot/agentic/arc_object_history_salience.py module docstring.
SUBMITTED_OBJECT_HISTORY_SALIENCE_ENABLED = False
SUBMITTED_PROGRAM_SYNTHESIS_PROPOSAL_FILTER_ENABLED = False
SUBMITTED_PROGRAM_SYNTHESIS_PROPOSAL_FILTER_TRUST_THRESHOLD = 0.75
SUBMITTED_AMORTIZED_FIRST_CONTACT_PRIOR_ENABLED = False
SUBMITTED_AMORTIZED_FIRST_CONTACT_PRIOR_MODE = (
    "frequency_prior_from_cross_game_first_contact_traces"
)
# Task 9 follow-on (2026-07-13): InertClickSigPruner is wired (rank_candidates filter + a real
# observe() hook from _ingest's per-transition OBSERVE site) but OFF by default, matching every
# other freshly-wired-but-unvalidated component in this file. Flipping this for the SCORED agent
# needs its own matched-budget offline A/B (states/actions-expanded reduction, zero regression in
# reproduced levels) per the solve_rate_dropped guardrail -- not assumed safe just because it's
# reachable. See python/carnot/agentic/arc_inert_click_pruner.py module docstring.
SUBMITTED_INERT_CLICK_PRUNER_ENABLED = False
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
SUBMITTED_EPISTEMIC_LEDGER_ENABLED = True
SUBMITTED_EPISTEMIC_LEDGER_MODE = "agent_owned_visible_state_hypothesis_ledger"
SUBMITTED_STRUCTURED_EVIDENCE_MEMORY_ENABLED = False
SUBMITTED_STRUCTURED_EVIDENCE_MEMORY_MODE = "agent_owned_event_tape_plus_identical_byte_index"
# E1 (arXiv:2512.24156, the hidden-leaderboard 3rd-place "just-explore" solver) status-bar
# masking. `StepwiseExplorer.hud_mask` has existed as a constructor param since before this
# flag (see `_hash`, which already collapses masked cells) but was never populated on the
# live path -- neither E3AgentPolicy nor CarnotAgentPolicy ever computed or passed one, so a
# ticking score/timer/step-counter HUD cell made every tick look like a brand-new state to
# the live dedup. Rule-based (`_compute_hud_mask_from_frame`, single observed frame, zero
# extra actions) rather than the probe-based `arc_graph_explore.discover_hud_mask` (burns up
# to 4 real actions from reset -- fine for the offline dev harness where resets are free, not
# viable under RHAE live scoring where extra actions are squared against the human baseline).
# 2026-07-12 RESOLUTION (REQ-ARC-WMTE-5583): two offline matched-budget A/Bs (exp5584,
# exp5585 -- the second using E3AgentPolicy's own REAL default search components, not a
# weak stand-in) both confirmed the mechanism's direct effect (distinct_states_delta:
# large, consistent, HUD-positive-games-only reductions; positive control clean both
# times) with ZERO measured harm on the in-roster HUD-negative control game, but both
# hit the SAME floor effect on levels_gained (no level-up reached in EITHER arm on ANY
# roster game at either budget) -- a limitation of the current from-scratch explorer's
# basic capability ceiling on this roster, independent of auto_hud_mask, confirmed by
# adversarial_verify.py's FALSE_NEGATIVE_RISK flag on both artifacts. Per operator
# directive: invoking the tier-3 LLM proposer to manufacture headroom for a third
# offline A/B was rejected (the verified hidden-leaderboard top-3 all use NO LLM
# proposer at all, per a comparative leaderboard analysis this session). The
# levels_gained question is DEFERRED to live-submission telemetry rather than further
# offline attempts; flipped ON here on the strength of the established safety case
# (structurally the mechanism can only collapse cells already proven action-invariant,
# never introduce a false collapse of genuine board state -- see
# _compute_hud_mask_from_frame's docstring), not a proven levels_gained win. See
# REQ-ARC-WMTE-5583's RESOLUTION note for the full record.
SUBMITTED_AUTO_HUD_MASK_ENABLED = True
SUBMITTED_AUTO_HUD_MASK_MODE = "rule_based_status_bar_classifier_single_frame"
# REQ-ARC-WMTE-5960: the REPAIRED status-bar classifier plus a runtime collapse guard on its
# application. Read python/carnot/agentic/arc_hud_bar_detector.py's module docstring for the
# full WHY; in short:
#
# THE DEFECT. The classifier above (`_compute_hud_mask_from_frame` ->
# `ColorBlobSaliencePrior.is_status_bar_like`) is ORIENTATION-BLIND, not merely mis-tuned: its
# geometric branch requires `width >= 0.75*frame_width AND height <= 2`, a horizontal-bar
# template no vertical bar can satisfy at any frame size, and its edge test reads bbox[0]/bbox[2]
# (both Y coordinates), so the LEFT/RIGHT columns are never tested at all. r11l renders a
# MONOTONE step counter into frame COLUMN 0 -- one 4-connected blob, colour 0, 64 px, bbox
# (0,0,63,0) -- so the mask resolves to None there, nothing leaves node identity, and the search
# has no memory: 1956 actions -> 1392 graph nodes over 31 true game states (44.9x inflation on
# ARM A; ~22-23x on the current live config, arm B2 -- cite the per-arm baseline), with a single
# WALL-blocked, game-state-INERT click re-popped 1371 of those 1956 actions because it changes
# exactly one cell, in column 0. Masking column 0 and changing NOTHING else makes r11l WIN on
# 3/3 seeds. That column-0 mask was ORACLE-DERIVED from the public game source and is a
# DIAGNOSTIC ONLY; the detector here derives the identical 64-cell mask from FRAME STATISTICS
# alone (all four edges with a symmetric 2-cell tolerance + a scale-free orientation-aware
# elongation ratio + a 5%-of-frame area ceiling). Verified on all 25 public games: a strict
# SUPERSET of today's mask (drops nothing, 25/25) and a strict SUBSET of the reference
# solver's mask (0 extra px, 25/25).
#
# WHY DEFAULT OFF. Over-masking destroys CORRECTNESS while under-masking only costs efficiency,
# and that asymmetry is not hypothetical: injecting the reference's mask into our own
# conservative application produced PROVEN aliasing violations on tu93 (2 of 58 observable keys,
# unmasked control 0 of 14) and lf52 -- every one of them a monotone counter GATING the
# game-over transition, i.e. a decision-relevant state variable hiding inside a textbook HUD.
# So the flip needs its own matched-budget per-seed full-corpus A/B (arm G vs arm B2 in
# experiment_5836), and the submitted agent stays byte-identical until then.
SUBMITTED_EDGE_BAR_HUD_MASK_ENABLED = False
SUBMITTED_EDGE_BAR_HUD_MASK_MODE = "orientation_complete_edge_bar_geometry_single_frame"
# The runtime HARD REFUSAL that makes the above safe to consider at all. A
# `(masked_node, concrete_action)` key observed to produce TWO DIFFERENT masked successors
# proves one masked hash covers two behaviourally distinct true states -- a causal proof from
# the agent's OWN transitions, no oracle. Each proof un-masks that node (local split); past
# HUD_MASK_GUARD_MAX_SPLIT_NODES the mask is revoked outright and identity falls back to
# unmasked. Carries a MANDATORY unmasked control so environment non-determinism is not
# misattributed to the mask (measured: sc25's masked violations are matched 1:1 by
# unmasked-control violations, so none of them is the mask's fault). Default OFF so it is
# flipped together with the detector it guards, never silently ahead of it.
SUBMITTED_HUD_MASK_COLLAPSE_GUARD_ENABLED = False
SUBMITTED_HUD_MASK_COLLAPSE_GUARD_MODE = "per_node_successor_branching_proof_with_unmasked_control"
# REQ-ARC-WMTE-5717: inject a small game-AGNOSTIC exploration-playbook few-shot (the recurring
# "orient, hypothesize, test, revise" method distilled from the solve corpus,
# docs/research-notes/arc-exploration-playbook-20260717.md) into the STALL/first-contact
# world-model re-induction prompt, to bias the tier-3 proposer's priors without long reasoning
# (exp5714 found long-reasoning induction overruns the budget). OFF by default (dev-gated,
# same pattern as SUBMITTED_COLOR_BLOB_SALIENCE_ENABLED): the operator graduates it only if the
# matched-budget A/B (exp5717) shows an induction-quality gain. Also togglable at runtime via
# CARNOT_ARC_PLAYBOOK_EXEMPLARS_ENABLED=1 for the A/B harness (the prompt-side gate in
# arc_executable_world_model.induce_prompt reads that env var, so BOTH the flag path here and
# the env path there must agree before any exemplar text is injected).
SUBMITTED_PLAYBOOK_EXEMPLARS_ENABLED = False
# REQ-ARC-WMTE-5836: the two just-explore (arXiv:2512.24156) FRONTIER-DISCIPLINE mechanisms,
# grafted from python/carnot/agentic/arc_frontier_discipline.py -- read that module's docstring
# for the full WHY. In one sentence each:
#   * TIER EXHAUSTION -- a strict GLOBAL priority barrier: no tier-N+1 action ANYWHERE in the
#     graph until every tier-<=N action EVERYWHERE has been tried. This is NOT the already-
#     measured-null CARNOT_ARC_TIER_SCHEDULE sort (that reorders within one node; this gates
#     across all nodes). The 5-tier predicate is shared with that sort deliberately so an A/B
#     difference cannot be a predicate difference.
#   * DISTANCE GRADIENT -- pick the frontier target that is navigation-NEAREST to where the
#     agent is standing (one multi-source reverse BFS over known-working edges), instead of
#     the shallowest-from-root node, which can be arbitrarily far away and costs a full
#     RESET+replay every time.
#   * TIER UNIFORM RANDOM -- draw uniformly among the tier-eligible candidates instead of
#     taking the first. The reference does this (random.choice); Carnot's live pop(0) is the
#     fully-greedy opposite, and three prior experiments found that replacing the reference's
#     uniform draw with ANY Carnot-scored order LOST solves. So the greedy draw is itself a
#     suspect and the A/B must be able to vary it independently of the barrier.
# FLIPPED ON 2026-07-25 by operator decision, after the A/B that these flags were gated on.
# EVIDENCE (results/experiment_5836_frontier_definitive.json, full declared spec: 7 arms x 25 games
# x 3 conditions = 975 cells, 0 errored, reproduction 29/30):
#   arm B2 = TIER_EXHAUSTION + TIER_UNIFORM_RANDOM + TIER_CLICK_VOCAB_ONLY
#     mean wins/cell  10.56  vs baseline 7.00   (+3.56 games)
#     strict PER-SEED dominance over baseline in 8 of 9 (condition x seed) cells
#     holds across real / recoloured / reflected: 10.00 / 10.67 / 11.00
#     its ONE loss is cd82, diagnosed as a budget-WALL artifact (solves at budget 4000; it used
#     1977 of 2000 actions), not lost capability -- see cd82_residual_diagnosis in
#     results/experiment_5836_frontier_click_vocab_gate.json
#   the click gate's contribution is isolated by B2 vs B2_nofix: dominance 8/9 vs 3/9, and tu93
#     lost in 0/9 vs 6/9 cells
#   the DISTANCE GRADIENT stays OFF: arm D (tier+gradient) scored 9.67 with dominance 2/3, worse
#     than B2 on both axes. Only the two mechanisms below earned the flip.
# HONEST LIMIT ON THIS DECISION: the evidence is PUBLIC-game, via the offline dev twin. The
# positive control (arm E, the just-explore reference through our shim) scored 6.67 -- BELOW our own
# baseline -- so by the A/B's own pre-registered rule the harness is the confound for any
# CROSS-IMPLEMENTATION claim, and none is made here. What justifies the flip is the INTERNAL
# comparison A vs B2 vs B2_nofix: the same live explorer differing only by flag, where a per-seed
# delta cannot be a cross-implementation artifact. This is a capability argument, NOT demonstrated
# hidden-game transfer.
SUBMITTED_FRONTIER_TIER_EXHAUSTION_ENABLED = True
SUBMITTED_FRONTIER_TIER_EXHAUSTION_MODE = "just_explore_global_priority_group_exhaustion"
SUBMITTED_FRONTIER_TIER_COUNT = FRONTIER_TIER_COUNT
SUBMITTED_FRONTIER_TIER_UNIFORM_RANDOM_ENABLED = True
# REQ-ARC-WMTE-5836 follow-up (2026-07-25): CONFINE the tier barrier to games whose action
# vocabulary actually contains CLICK, discovered at RUNTIME from frame.available_actions.
#
# WHY. The full-spec A/B (results/experiment_5836_frontier_discipline_generalization.json) found a
# real, recolour- and reflection-robust capability gain (+2..+4 games in every condition) but NO arm
# was regression-free per seed. Arm B2's ONLY loss was tu93 -- the single nav-only game among the
# baseline's wins -- and it lost it on 2 of 3 seeds in EVERY condition, while all five of its GAINS
# were click games. That asymmetry is not a coincidence: the 5-tier predicate ranks CLICK-TARGET
# salience (button-like vs status-bar vs large-flat blobs). On a nav-only game there are no click
# targets to rank, so the barrier cannot express anything useful there -- it can only perturb the
# move ordering of a search that was already succeeding. Gating on click availability is therefore
# the mechanism's correct DOMAIN OF DEFINITION, not a post-hoc carve-out to rescue a number.
#
# HIDDEN-GAME LEGAL. The signal is frame.available_actions, which the env reports at runtime on any
# game including one never seen before -- parsed with the existing _available_action_ids() helper
# (handles enums, "ACTION6" strings and bare ints). It is NOT the harness's hardcoded CLICK_GAMES
# list, which would be per-game knowledge and illegal for a hidden game.
#
# Defaults TRUE because it only takes effect where the barrier is already enabled, and where it
# takes effect the barrier was measurably harmful. Set False to reproduce the pre-fix behaviour.
SUBMITTED_FRONTIER_TIER_CLICK_VOCAB_ONLY_ENABLED = True
SUBMITTED_FRONTIER_DISTANCE_GRADIENT_ENABLED = False
SUBMITTED_FRONTIER_DISTANCE_GRADIENT_MODE = "multi_source_reverse_bfs_over_known_working_edges"

# REQ-ARC-WMTE-5950: the just-explore CLICK-TARGET GENERATION rule -- a uniform random
# pixel OF the chosen object instead of that object's truncated centroid, redrawn on
# revisit. DEFAULT OFF: the submitted agent is byte-identical until a matched-budget
# offline A/B greenlights a flip (arm F vs arm B2 in experiment_5836, where B2 is the
# CURRENT live configuration -- NOT arm A, which pins the pre-flip flags explicitly).
#
# WHY IT IS A CANDIDATE AT ALL (see arc_component_sampling.py for the full write-up):
# the truncated centroid is not a member of its own object on 100% of 204 measured real
# r11l states (mean 5.94 such objects per state), truncated centroids collide and get
# de-duplicated away (r11l 37 objects -> 34 click rows), and on games where the click
# coordinate PARAMETERISES the move (r11l's handle drag) one fixed point per object
# collapses a pixel-continuous action space to a few dozen frozen coordinates.
#
# WHAT IT IS NOT: it is NOT the r11l fix. r11l's measured defect is state-identity
# aliasing (its step counter renders into frame column 0, auto_hud_mask resolves to None
# there, 44.9x node inflation, one inert click re-popped 1371/1956 actions). A winning
# 3-click r11l sequence already exists inside today's candidate set. Any artifact must
# therefore gate this mechanism on a FULL-CORPUS regression result, never on r11l.
SUBMITTED_CLICK_PIXEL_SAMPLING_ENABLED = False
SUBMITTED_CLICK_PIXEL_SAMPLES_PER_COMPONENT = 1
# Bounded WITH-REPLACEMENT budget. The reference redraws indefinitely (it retires an
# object only when a click on it produced NO state change), which our drain-only
# ``node["untested"]`` list cannot express at all. This budget is the minimal change that
# permits revisiting: at most N total draws per (node, object). N=1 => no redraw at all,
# i.e. pure one-shot-per-object with a corrected coordinate.
SUBMITTED_CLICK_PIXEL_REDRAW_BUDGET = 3


def _playbook_exemplars_gate_on() -> bool:
    """REQ-ARC-WMTE-5717: the DEV-ONLY gate for playbook-exemplar injection -- the SUBMITTED
    module flag OR the CARNOT_ARC_PLAYBOOK_EXEMPLARS_ENABLED runtime env override (the same
    flag-OR-env pattern as the SGE candidate router). Stall-only scoping is applied at the
    call site; this only answers "is the feature gated on at all"."""
    import os as _os

    return bool(SUBMITTED_PLAYBOOK_EXEMPLARS_ENABLED) or (
        _os.environ.get("CARNOT_ARC_PLAYBOOK_EXEMPLARS_ENABLED") == "1"
    )


# REQ-ARC-WMTE-5718: RETRIEVAL (graph-RAG) variant of the playbook injection. Instead of the
# fixed exemplar block, embed the CURRENT stuck situation with the loaded model and inject ONLY
# the top-K patterns relevant to it (from the offline models/arc_playbook_index/). OFF by default;
# when on it TAKES PRECEDENCE over the static block on the stall path (falling back to the static
# block, then to nothing, if retrieval is unavailable). Dev-gated, never touches the frozen submit.
SUBMITTED_PLAYBOOK_RETRIEVAL_ENABLED = False
PLAYBOOK_RETRIEVAL_TOPK = 4


def _playbook_retrieval_gate_on() -> bool:
    """REQ-ARC-WMTE-5718: the DEV-ONLY gate for RETRIEVAL-based injection -- the SUBMITTED module
    flag OR the CARNOT_ARC_PLAYBOOK_RETRIEVAL runtime env override."""
    import os as _os

    return bool(SUBMITTED_PLAYBOOK_RETRIEVAL_ENABLED) or (
        _os.environ.get("CARNOT_ARC_PLAYBOOK_RETRIEVAL") == "1"
    )


_DEFAULT_VALUE_HEAD = object()
_DEFAULT_CANDIDATE_ROUTER = object()
_DEFAULT_FRAME_CHANGE_SCORER = object()
_DEFAULT_GOAL_BIAS = object()
_DEFAULT_EPISTEMIC_LEDGER = object()
_DEFAULT_STRUCTURED_EVIDENCE_MEMORY = object()
# REQ-ARC-FCP-5703 / GAP-5703: thresholds for goal_bias_diagnostics()'s degenerate-score
# self-audit. Mirrors GoalEnergyCandidateGuidance's arms_non_degenerate variance floor
# (arc_goal_energy_live.py: `variance > 1e-12`); the minimum-sample floor guards against a
# false "degenerate" read on a short episode that simply has not scored enough nodes yet.
_GOAL_BIAS_DEGENERACY_MIN_SAMPLES = 20
_GOAL_BIAS_DEGENERACY_VARIANCE_EPS = 1e-12


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


def _compute_hud_mask_from_frame(frame: Any, *, edge_bar_detector: bool = False) -> Any | None:
    """Rule-based, zero-action-cost status-bar mask for `StepwiseExplorer.hud_mask`.

    REQ-ARC-WMTE-5960: when `edge_bar_detector` is True the mask comes from the REPAIRED,
    orientation-complete detector in `carnot.agentic.arc_hud_bar_detector` instead. That
    detector ORs in this function's existing `is_status_bar_like` predicate, so its output is
    a SUPERSET of this one BY CONSTRUCTION -- an A/B difference can therefore only come from
    newly-detected cells, never from cells that silently stopped being masked. The default
    (False) leaves this function byte-identical to its pre-5960 behaviour. See
    SUBMITTED_EDGE_BAR_HUD_MASK_ENABLED above for the measured defect and why the repair
    ships default-off.

    E1 (arXiv:2512.24156): segment the frame into single-color connected components and
    mark any status-bar-LIKE blob's cells (frame-edge-touching, spans most of the frame's
    width, thin) as HUD, using the SAME geometric rule
    `ColorBlobSaliencePrior.is_status_bar_like` already applies to deprioritize status-bar
    click candidates -- just reused here to also collapse those cells out of node identity.
    Computed from a SINGLE already-observed frame (no env access, no probe actions), unlike
    `arc_graph_explore.discover_hud_mask` which burns up to 4 real actions from reset and is
    only viable in the offline dev harness. Returns a bool grid-shaped mask (True = HUD
    cell) or None if nothing looked like a status bar (a safe no-op: `_hash` leaves the grid
    untouched when `hud_mask` is None).
    """

    if frame is None:
        return None
    try:
        import numpy as np
        from carnot.agentic.arc_agi3_world_model import grid_of

        grid = grid_of(frame)
        if getattr(grid, "shape", None) is None or grid.ndim != 2:
            return None
        if edge_bar_detector:
            return compute_edge_bar_hud_mask(grid, include_status_bar_like=True)
        prior = ColorBlobSaliencePrior()
        mask = np.zeros(grid.shape, dtype=bool)
        for blob in connected_color_blobs(grid):
            if prior.is_status_bar_like(blob):
                for y, x in blob.cells:
                    mask[y, x] = True
        return mask if bool(mask.any()) else None
    except Exception:
        return None


def _route_explore_budget(route: dict[str, Any]) -> int:
    if route.get("uses_goal_distance_heuristic") is False:
        return SUBMITTED_ROUTED_EXPLORE_BUDGET
    return SUBMITTED_GRAPH_EXPLORE_BUDGET


def _recommend_live_approach(
    game_id: str,
    *,
    mechanic: Optional[str] = None,
    early_play_signature: Any = None,
) -> dict[str, Any]:
    """Return the solve-learning recommendation, falling back to the lightweight strategy router.

    `early_play_signature` (a list of {action_id, data, before, after} rows built purely from this
    game's OWN observed transitions, no game-identity lookup) threads through to
    `arc_solve_learning.recommend_approach`'s behavioral `feature_router` payload — this is the
    genre/mechanic-class-from-play-experience signal, distinct from (and available even when) the
    identity-keyed `mechanic=` hint and the games-registry similarity engine are both blind to a
    truly hidden game.
    """

    try:
        rec = arc_solve_learning.recommend_approach(
            game_id, mechanic=mechanic, early_play_signature=early_play_signature
        )
        if isinstance(rec, dict) and isinstance(rec.get("strategy"), dict):
            return rec
    except Exception as exc:
        return {
            "error": f"recommend_approach_failed:{type(exc).__name__}",
            "strategy": arc_strategy_router.route_for_game(game_id, mechanic=mechanic),
        }
    return {"strategy": arc_strategy_router.route_for_game(game_id, mechanic=mechanic)}


# REQ-CAPSTONE-4582 (live wiring): how many of the game's OWN early observed transitions to
# summarize into a behavioral mechanic-class signature before biasing the live search. Matches
# `arc_solve_learning.extract_early_play_signature`'s own default `k`.
_FEATURE_ROUTER_EARLY_PLAY_K = 8

# Minimum `feature_router.confidence` (learned win-rate margin for the routed approach, from
# `arc_solve_learning.learn_feature_router_policy`) required before the classified mechanic is
# allowed to change live search behavior. Below this it is observational only (still stored on
# `self.feature_router`). Conservative given exp4582's null on the closely-related full-solver-swap
# version of this idea -- see `_maybe_route_from_transitions`'s docstring.
_FEATURE_ROUTER_MIN_CONFIDENCE = 0.5

# ------------------------------------------------------------------------------------------------
# RUN-LOCAL cross-game adaptation (scoped: docs/research-notes/arc-agi3-run-local-cross-game-
# adaptation-scope-2026-07-12.md; operator-cleared to build per that doc's recorded ruling).
#
# One Kaggle submission's Swarm.main() instantiates every game's agent up front and runs them all
# CONCURRENTLY on separate threads within ONE process (confirmed against both the vendored
# ARC-AGI-3-Agents reference and scripts/kaggle/submission_kernel/main.py's real competition-rerun
# path -- one gateway, one scorecard, one process, for the whole game roster). This ledger lets
# games within that SAME run share a cheap, thread-safe, NEVER-persisted-to-disk signal about which
# (mechanic_class, approach) pairs have been making progress so far -- learning a class of games'
# behavior across a run, never a memorized per-game action sequence (the operator's own dividing
# line). It NEVER touches ops/arc_solve_registry.yaml and never survives past the process.
#
# Ships OFF by default (CARNOT_ARC_RUN_LOCAL_ADAPTATION=1 to opt in). Per the scope doc's own Phase
# Prototype + Empirical Validation + Adversarial Check discipline, this component is the prototype
# only -- the offline concurrent-multi-game-sequence A/B measurement (scope doc SS2.4) has NOT run
# yet, so do not flip the default on without it.
# ------------------------------------------------------------------------------------------------

# In-run minimum sample count (distinct game instances) for a mechanic class before its recorded
# outcomes are trusted to nudge the confidence gate at all. Deliberately small (a submission roster
# is not 30 games of one mechanic class) -- this is an explicitly WEAK signal, not a statistically
# significant one; treat any measured effect from it with the same skepticism the scope doc applies.
_RUN_LOCAL_MIN_SAMPLES = 3

# Maximum confidence bonus this mechanism can contribute, added to (never replacing)
# `feature_router.confidence` before the `_FEATURE_ROUTER_MIN_CONFIDENCE` gate. Bounded and small on
# purpose: worst case, behavior converges to the already-shipped per-game-only default.
_RUN_LOCAL_MAX_CONFIDENCE_BONUS = 0.2


class RunLocalMechanicLedger:
    """Thread-safe, in-process, NEVER-persisted ledger of (mechanic_class, approach) outcomes for
    games completed so far in THIS run. See the module-level comment above for the concurrency
    rationale. Keyed per-game-instance (not appended) so a still-in-progress game's repeated updates
    overwrite its own prior entry rather than accumulating duplicates."""

    def __init__(self) -> None:
        import threading

        self._lock = threading.Lock()
        self._records: dict[str, dict[int, tuple[str, float]]] = {}

    def update(
        self, mechanic_class: str, game_key: int, approach: str, outcome_score: float
    ) -> None:
        if not mechanic_class or mechanic_class == "unknown" or not approach:
            return
        with self._lock:
            self._records.setdefault(mechanic_class, {})[game_key] = (
                approach,
                float(outcome_score),
            )

    def sample_count(self, mechanic_class: str) -> int:
        with self._lock:
            return len(self._records.get(mechanic_class, {}))

    def mean_outcome(self, mechanic_class: str, approach: str) -> Optional[float]:
        with self._lock:
            rows = list(self._records.get(mechanic_class, {}).values())
        matched = [score for a, score in rows if a == approach]
        if not matched:
            return None
        return sum(matched) / len(matched)


# Module-level singleton: one ledger per process, shared by every concurrently-running game's
# E3AgentPolicy instance in the same Swarm run (see the concurrency rationale above).
_RUN_LOCAL_LEDGER = RunLocalMechanicLedger()


def _run_local_adaptation_enabled() -> bool:
    import os

    return os.environ.get("CARNOT_ARC_RUN_LOCAL_ADAPTATION") == "1"


def _run_local_confidence_bonus(mechanic_class: str, approach: str) -> float:
    """A small, bounded confidence bonus from OTHER games' in-run evidence for this exact
    (mechanic_class, approach) pair -- 0.0 (no effect) unless this run has already accumulated
    `_RUN_LOCAL_MIN_SAMPLES` completed-or-in-progress games of this mechanic class. Never consults
    game identity; purely a function of what this SAME process has observed so far this run."""

    if not _run_local_adaptation_enabled():
        return 0.0
    if _RUN_LOCAL_LEDGER.sample_count(mechanic_class) < _RUN_LOCAL_MIN_SAMPLES:
        return 0.0
    mean = _RUN_LOCAL_LEDGER.mean_outcome(mechanic_class, approach)
    if mean is None:
        return 0.0
    return max(0.0, min(_RUN_LOCAL_MAX_CONFIDENCE_BONUS, mean * _RUN_LOCAL_MAX_CONFIDENCE_BONUS))


def _run_local_outcome_proxy(transitions: Any, *, k: int = _FEATURE_ROUTER_EARLY_PLAY_K) -> float:
    """Cheap, HONEST-ABOUT-BEING-A-PROXY early-progress signal: 1.0 if any of this game's first `k`
    transitions advanced the level, else 0.0. This is NOT the competition's real RHAE scoring
    formula -- it is a scaffold for the prototype ledger above. The scope doc's SS2.4 validation
    harness should measure against the real scoring formula for its own headline claim; this proxy
    only needs to be cheap and directionally sane for the live nudge to be well-defined."""

    for transition in list(transitions or [])[:k]:
        level_before = getattr(transition, "level_before", None)
        level_after = getattr(transition, "level_after", None)
        if level_before is not None and level_after is not None and level_after > level_before:
            return 1.0
    return 0.0


def _early_play_rows(
    transitions: Any, *, k: int = _FEATURE_ROUTER_EARLY_PLAY_K
) -> list[dict[str, Any]]:
    """Convert this game's own collected `Transition`s into the row shape
    `arc_solve_learning.extract_early_play_signature` expects. Purely behavioral: uses only the
    grids/actions this specific live play session has already observed, never a game-identity
    lookup and never the executable win-check."""

    rows: list[dict[str, Any]] = []
    for transition in list(transitions)[:k]:
        rows.append(
            {
                "action_id": getattr(transition, "action", None),
                "data": getattr(transition, "data", None),
                "before": getattr(transition, "grid", None),
                "after": getattr(transition, "next_grid", None),
            }
        )
    return rows


def _load_sge_candidate_router(game_id: str) -> Any | None:
    """REQ-ARC-FCP-5699-11: build the LLM Strategy-Guided Exploration router, reusing the
    frozen live-submission generator (Qwen3.5-9B-MTP) via a LocalGGUFProposer configured
    IDENTICALLY to _proposer()'s own lazy default (same repo_substr/mtp/kv_quant/
    no_think_prefix/model_path env/n_gpu_layers env) -- LocalGGUFProposer._ensure_server()
    reuses ANY already-healthy server on the configured port regardless of which call built
    it first, so this and the induction proposer share ONE warm server, never a second
    model load. Returns None on any failure (caller falls back to the discriminative
    router) -- this must never break the live path just because SGE construction failed."""
    import os as _os

    from carnot.agentic.arc_executable_world_model import LocalGGUFProposer
    from carnot.agentic.arc_llm_strategy_proposer import LLMStrategyProposer, SGECandidateRouter

    gguf = LocalGGUFProposer(
        repo_substr="Qwen3.5-9B-MTP",
        model_path=_os.environ.get("CARNOT_ARC_GGUF_PATH") or None,
        mtp=(_os.environ.get("CARNOT_ARC_MTP", "1") != "0"),
        kv_quant="q8_0",
        no_think_prefix="/no_think\n",
        # REQ-ARC-FCP-5699-35: read the SAME env-var-overridable defaults as _proposer()'s own
        # lazy default (4096/600, was a hardcoded max_tokens=2560 literal with no timeout
        # override at all -- caught by test_req_arc_fcp_5699_11_load_sge_candidate_router_
        # reuses_frozen_generator_config's own "configured IDENTICALLY" contract when the
        # _proposer() default graduated and this one silently didn't move with it).
        max_tokens=int(_os.environ.get("CARNOT_ARC_INDUCE_MAX_TOKENS", "4096")),
        timeout=int(_os.environ.get("CARNOT_ARC_INDUCE_TIMEOUT", "600")),
        n_gpu_layers=int(_os.environ.get("CARNOT_ARC_NGL", "999")),
    )
    return SGECandidateRouter(
        proposer=LLMStrategyProposer(completer=gguf, max_tokens=64),
        game_id=game_id,
        k=3,
        temperatures=(0.3, 0.6, 0.9),
        max_candidates=8,
        reflect_every=6,
    )


def _sge_candidate_router_requested() -> bool:
    """REQ-ARC-FCP-5699-12: env-var escape hatch for A/B measurement, mirroring
    CARNOT_ARC_DISABLE_INDUCTION's pattern -- the module-level SUBMITTED_SGE_CANDIDATE_
    ROUTER_ENABLED flag stays the single source of truth for what SHIPS, but a subprocess-
    based measurement (scripts/kaggle/arc_local_submission_gate.py spawns one fresh
    Python process per game) cannot monkeypatch an in-process module attribute -- an env
    var is the only way to flip this for a single measurement run without touching the
    committed default."""
    import os as _os

    return (
        bool(SUBMITTED_SGE_CANDIDATE_ROUTER_ENABLED)
        or _os.environ.get("CARNOT_ARC_SGE_CANDIDATE_ROUTER") == "1"
    )


def _load_submitted_candidate_router(game_id: str = "unknown_game") -> Any | None:
    """Load the live candidate router. Default: the Exp4545 v3 discriminative router (a
    safe candidate-order tie-breaker). REQ-ARC-FCP-5699-11: when
    SUBMITTED_SGE_CANDIDATE_ROUTER_ENABLED (or CARNOT_ARC_SGE_CANDIDATE_ROUTER=1, for
    measurement runs), tries the SGE router first, falling through to the discriminative
    router (never None-ing out the live path) if SGE construction fails for any reason --
    default False, so behavior is unchanged unless explicitly opted in."""

    if _sge_candidate_router_requested():
        try:
            sge_router = _load_sge_candidate_router(game_id)
            if sge_router is not None:
                return sge_router
        except Exception:
            pass
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
    """REQ-ARC-WMTE-4640/5711: load submitted visible-state goal energy."""

    if not SUBMITTED_GOAL_ENERGY_ENABLED:
        return None
    return arc_goal_energy_live.load_relational_goal_energy(root=REPO)


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
        auto_hud_mask: bool = SUBMITTED_AUTO_HUD_MASK_ENABLED,
        edge_bar_hud_mask: bool | None = None,
        hud_mask_collapse_guard: bool | None = None,
        hud_mask_guard_max_split_nodes: int = HUD_MASK_GUARD_MAX_SPLIT_NODES,
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
        tier_exhaustion: bool | None = None,
        tier_count: int = SUBMITTED_FRONTIER_TIER_COUNT,
        tier_uniform_random: bool | None = None,
        tier_click_vocab_only: bool | None = None,
        frontier_gradient: bool | None = None,
        frontier_discipline_seed: int = 20260724,
        click_pixel_sampling: bool | None = None,
        click_pixel_samples_per_component: int = SUBMITTED_CLICK_PIXEL_SAMPLES_PER_COMPONENT,
        click_pixel_redraw_budget: int = SUBMITTED_CLICK_PIXEL_REDRAW_BUDGET,
        click_pixel_sampling_seed: int | None = None,
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
        inert_click_pruner: Any | bool | None = SUBMITTED_INERT_CLICK_PRUNER_ENABLED,
        amortized_first_contact_prior: Any | bool | None = (
            SUBMITTED_AMORTIZED_FIRST_CONTACT_PRIOR_ENABLED
        ),
        go_explore_archive: Any | bool | None = SUBMITTED_GO_EXPLORE_ARCHIVE_ENABLED,
        similarity_retrieval: bool | None = None,
        similarity_bucket_width: float = MATM_SIMILARITY_BUCKET_WIDTH,
        similarity_max_candidates: int = MATM_SIMILARITY_MAX_CANDIDATES,
        transition_cycle_verifier: Any | None = None,
        generic_causal_primitive: Any | None = None,
        epistemic_ledger: Any | bool | None = None,
        structured_evidence_memory: Any | bool | None = None,
    ) -> None:
        self.hud_mask = hud_mask  # E1: mask step-counter cells out of node identity
        self.auto_hud_mask = bool(auto_hud_mask)
        self._hud_mask_attempted = hud_mask is not None
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
        self._last_action_salience_diagnostics: dict[str, Any] = {
            "connected_component_salience_enabled": False,
            "salience_tiers_emitted": False,
            "generation_stage_action_prioritization": False,
            "tier_rows": [],
            "action_tier_rows": [],
        }
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
        # REQ-ARC-WMTE-5836: the two just-explore frontier-discipline mechanisms. Resolution
        # order is explicit-kwarg > env override > SUBMITTED_* default, matching the
        # CARNOT_ARC_EXPLORE_DIVERSITY pattern below. The env path exists so the A/B harness can
        # flip an arm without mutating module globals (which would leak across arms inside one
        # process); the kwarg path exists so a test can construct a specific arm directly.
        import os as _fd_os

        def _fd_gate(explicit: bool | None, env_name: str, default: bool) -> bool:
            if explicit is not None:
                return bool(explicit)
            raw = _fd_os.environ.get(env_name)
            if raw is not None:
                return raw not in ("", "0", "false", "False")
            return bool(default)

        self.tier_exhaustion_enabled = _fd_gate(
            tier_exhaustion,
            "CARNOT_ARC_FRONTIER_TIER_EXHAUSTION",
            SUBMITTED_FRONTIER_TIER_EXHAUSTION_ENABLED,
        )
        self.tier_uniform_random_enabled = _fd_gate(
            tier_uniform_random,
            "CARNOT_ARC_FRONTIER_TIER_UNIFORM_RANDOM",
            SUBMITTED_FRONTIER_TIER_UNIFORM_RANDOM_ENABLED,
        )
        self.tier_click_vocab_only = _fd_gate(
            tier_click_vocab_only,
            "CARNOT_ARC_FRONTIER_TIER_CLICK_VOCAB_ONLY",
            SUBMITTED_FRONTIER_TIER_CLICK_VOCAB_ONLY_ENABLED,
        )
        # Runtime-discovered: has THIS game ever offered a click action? Never per-game knowledge.
        self._fd_click_vocab_seen = False

        self.frontier_gradient_enabled = _fd_gate(
            frontier_gradient,
            "CARNOT_ARC_FRONTIER_GRADIENT",
            SUBMITTED_FRONTIER_DISTANCE_GRADIENT_ENABLED,
        )
        # REQ-ARC-WMTE-5960: the repaired HUD detector and its runtime collapse guard, resolved
        # through the SAME explicit-kwarg > env > SUBMITTED_* ladder as everything above so the
        # A/B harness can flip one arm without mutating module globals across arms in-process.
        self.edge_bar_hud_mask_enabled = _fd_gate(
            edge_bar_hud_mask,
            "CARNOT_ARC_EDGE_BAR_HUD_MASK",
            SUBMITTED_EDGE_BAR_HUD_MASK_ENABLED,
        )
        self.hud_mask_collapse_guard_enabled = _fd_gate(
            hud_mask_collapse_guard,
            "CARNOT_ARC_HUD_MASK_COLLAPSE_GUARD",
            SUBMITTED_HUD_MASK_COLLAPSE_GUARD_ENABLED,
        )
        self._hud_collapse_guard = (
            MaskCollapseGuard(max_split_nodes=int(hud_mask_guard_max_split_nodes))
            if self.hud_mask_collapse_guard_enabled
            else None
        )
        # Recorded for the artifact row: WHICH detector produced the mask that is actually in
        # force, so a reader never has to infer the treatment from a cell count.
        self._hud_mask_source = "explicit_kwarg" if hud_mask is not None else "unresolved"
        # Distinct UNMASKED frame hashes observed. The denominator of the dedup metric: with a
        # working mask, many distinct raw frames share one graph node; with none, the two counts
        # are equal (measured on all 8 mask-None public games) and the search has no memory.
        self._unique_frame_hashes: set[str] = set()
        # Unmasked hash of the most recently observed frame -- the collapse guard's control
        # antecedent. Tracked here rather than read off the graph node because node frames are
        # only retained when certain optional components are enabled (see _ingest).
        self._last_unmasked_hash: Optional[str] = None
        self.tier_count = max(1, int(tier_count))
        self.tier_policy = TierExhaustionPolicy(tier_count=self.tier_count)
        self._active_tier = 0
        self._tier_advances = 0
        self._tier_deferrals = 0
        # Reverse index (target -> [(action, origin)]) over KNOWN-WORKING edges only, maintained
        # incrementally alongside self.adj because the gradient's multi-source BFS walks backward.
        # See arc_frontier_discipline.reverse_adjacency for the reference definition it must match.
        self.radj: dict[str, list] = {}
        self._gradient_targets = 0
        self._gradient_misses = 0
        # Pick-kind split for MECHANISM (b). An adversarial review (2026-07-24) found that a bare
        # "gradient fired N times" counter cannot distinguish the two very different things the
        # gradient can do, and that on deep-graph games the counter was 100% the degenerate case:
        # the gradient was returning self.cur (distance 0, because cur was one of its own seeds)
        # at a moment when the depth-first ride had ALREADY been refused by the max_depth cap --
        # so the "gradient" was really just cancelling the backtrack cap. These counters make that
        # visible in the artifact instead of hiding it inside one aggregate number.
        self._gradient_pick_cur = 0  # target == self.cur, cur legitimately under the depth cap
        self._gradient_pick_other = 0  # target is some OTHER node -- the real mechanism
        self._gradient_cur_at_cap_excluded = 0  # cur was depth-capped and refused as a seed
        self._fd_rng = __import__("random").Random(int(frontier_discipline_seed))
        # Own knob for the within-tier UNIFORM draw. DEFAULT None = draw uniformly over EVERY
        # tier-admitted row, which is the reference's `random.choice(untested_edges)`. Set
        # CARNOT_ARC_FRONTIER_TIER_DRAW_TOPK=<n> only to deliberately measure a top-n restriction
        # as a SEPARATE arm; it is a documented deviation from the reference, not the default.
        _fd_topk_raw = _fd_os.environ.get("CARNOT_ARC_FRONTIER_TIER_DRAW_TOPK")
        try:
            self._fd_draw_topk: int | None = (
                int(_fd_topk_raw) if _fd_topk_raw not in (None, "") else None
            )
        except ValueError:
            self._fd_draw_topk = None
        # REQ-ARC-WMTE-5950 -- per-object click-pixel sampling. Same explicit-kwarg > env >
        # SUBMITTED_* resolution as the frontier-discipline flags above.
        self.click_pixel_sampling_enabled = _fd_gate(
            click_pixel_sampling,
            "CARNOT_ARC_CLICK_PIXEL_SAMPLING",
            SUBMITTED_CLICK_PIXEL_SAMPLING_ENABLED,
        )
        self.click_pixel_samples_per_component = max(1, int(click_pixel_samples_per_component))
        self.click_pixel_redraw_budget = max(1, int(click_pixel_redraw_budget))
        # A SEPARATE RNG stream from self._fd_rng, deliberately. Sharing one would couple the
        # coordinate arm to the within-tier-draw arm: flipping the sampler would shift every
        # subsequent tier draw, so an A/B could not attribute a delta to either mechanism. The
        # stream is still SEEDED (unlike the reference, which reseeds from wall-clock and is
        # reproducible by nobody) so an arm remains re-runnable.
        self._cps_rng = __import__("random").Random(
            int(
                click_pixel_sampling_seed
                if click_pixel_sampling_seed is not None
                else frontier_discipline_seed
            )
            ^ 0x5A17
        )
        # NOT AN ACTIVITY COUNTER -- read the name literally. This counts click rows PRESENT
        # in the returned candidate list on calls made while the flag was on. It is identical
        # for a working sampler and for a totally dead one, which is exactly how a dead
        # mechanism previously reported itself as active-and-error-free. The activity
        # counter is `_cps_coords_changed` below.
        self._cps_rows_sampled = 0  # click rows PRESENT (not replaced) -- see comment above
        self._cps_redraws = 0  # bounded WITH-REPLACEMENT re-appends actually issued
        self._cps_redraws_declined_budget = 0  # pops that hit the per-(node, object) budget
        self._cps_redraws_declined_unresolved = 0  # coordinate not attributable to an object
        self._cps_redraws_declined_no_frame = 0  # node kept no frame -> cannot resolve
        self._cps_errors = 0  # exceptions swallowed inside the REDRAW half of the mechanism
        # THE ACTIVITY WITNESS (2026-07-25). Accumulated from the GENERATION path's own
        # SamplingDiagnostics, which used to be discarded. `coords_changed == 0` while the
        # flag is on means the mechanism replaced nothing -- it is a control, not a treatment
        # -- and `gen_errors > 0` says why. Without these two numbers there is NO field
        # anywhere evidencing that the generation rule fired at all, which made a one-shot
        # arm (redraw_budget=1, hence zero redraws by design) unfalsifiably indistinguishable
        # from a silent no-op.
        self._cps_coords_changed = 0
        self._cps_gen_errors = 0
        self._cps_points_in = 0
        self._cps_points_out = 0
        self._cps_unresolved = 0
        self._cps_contested_centroid_points = 0
        self.candidate_router = candidate_router
        self.epistemic_ledger = coerce_epistemic_ledger(epistemic_ledger)
        self.structured_evidence_memory = coerce_structured_evidence_memory(
            structured_evidence_memory
        )
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
        # REQ-ARC-FCP-5703 / GAP-5703: streaming (not stored-list, to avoid unbounded memory on
        # long episodes) mean/variance/min/max so goal_bias_diagnostics() can self-audit whether
        # this source is degenerate (constant score) on the current game -- mirrors the
        # arms_non_degenerate audit GoalEnergyCandidateGuidance already has. Observability only:
        # this does NOT disable goal_bias mid-episode, it only surfaces the finding.
        self._goal_bias_score_sum = 0.0
        self._goal_bias_score_sumsq = 0.0
        self._goal_bias_score_min: float | None = None
        self._goal_bias_score_max: float | None = None
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
        self.inert_click_pruner = coerce_inert_click_pruner(inert_click_pruner)
        self.generic_causal_primitive = coerce_generic_causal_primitive(generic_causal_primitive)
        self.amortized_first_contact_prior = coerce_amortized_first_contact_prior(
            amortized_first_contact_prior
        )
        self.go_explore_archive = coerce_go_explore_archive(go_explore_archive)
        self.transition_cycle_verifier = transition_cycle_verifier
        self._transition_cycle_receipts: list[dict[str, Any]] = []
        self._transition_cycle_admitted = 0
        self._transition_cycle_rejected = 0
        self._transition_cycle_abstained = 0
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
        """REQ-ARC-FCP-5703: `score_variance`/`degenerate` are a self-audit mirroring
        GoalEnergyCandidateGuidance's `arms_non_degenerate` check -- a goal_bias source
        that returns the same constant score on every real node (e.g. because its
        underlying frame-state extraction cannot parse this game's visual encoding, per
        GAP-5703's sp80 finding) mathematically cannot influence frontier ordering. This
        is observability only: `degenerate=True` does not disable goal_bias mid-episode."""

        n = int(self._goal_bias_scored)
        variance = 0.0
        if n > 0:
            mean = self._goal_bias_score_sum / n
            variance = max(0.0, self._goal_bias_score_sumsq / n - mean * mean)
        degenerate = bool(
            n >= _GOAL_BIAS_DEGENERACY_MIN_SAMPLES
            and variance <= _GOAL_BIAS_DEGENERACY_VARIANCE_EPS
        )
        return {
            "enabled": self.goal_bias is not None,
            "label": self.goal_bias_label,
            "lower_is_better": bool(self.goal_bias_lower_is_better),
            "nodes_scored": n,
            "errors": int(self._goal_bias_errors),
            "score_variance": round(variance, 10),
            "score_min": self._goal_bias_score_min,
            "score_max": self._goal_bias_score_max,
            "degenerate": degenerate,
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

    def transition_cycle_diagnostics(self) -> dict[str, Any]:
        """REQ-ARC-WMTE-5619: expose forward/inverse update-admission receipts."""

        if self.transition_cycle_verifier is None:
            return {"enabled": False}
        diagnostics: dict[str, Any] = {"enabled": True}
        if hasattr(self.transition_cycle_verifier, "diagnostics"):
            try:
                diagnostics.update(dict(self.transition_cycle_verifier.diagnostics()))
            except Exception:
                diagnostics["verifier_diagnostics_error"] = True
        diagnostics.update(
            {
                "admitted_update_count": int(self._transition_cycle_admitted),
                "rejected_update_count": int(self._transition_cycle_rejected),
                "abstained_update_count": int(self._transition_cycle_abstained),
                "immutable_update_receipts": list(self._transition_cycle_receipts),
            }
        )
        return diagnostics

    @staticmethod
    def _salience_diagnostic_prior(action_prior: Any | None) -> Any | None:
        prior = action_prior
        if prior is not None and not hasattr(prior, "tier_rows"):
            prior = getattr(prior, "base_prior", None)
        return prior if prior is not None and hasattr(prior, "tier_rows") else None

    def _record_action_salience_diagnostics(
        self,
        frame: Any,
        candidates: Sequence[Any],
        action_prior: Any | None,
    ) -> None:
        prior = self._salience_diagnostic_prior(action_prior)
        if prior is None:
            self._last_action_salience_diagnostics = {
                "connected_component_salience_enabled": False,
                "salience_tiers_emitted": False,
                "generation_stage_action_prioritization": False,
                "tier_rows": [],
                "action_tier_rows": [],
            }
            return
        try:
            tier_rows = list(prior.tier_rows(frame))
        except Exception:
            tier_rows = []
        try:
            action_tier_rows = (
                list(prior.action_tier_rows(frame, candidates))
                if hasattr(prior, "action_tier_rows")
                else []
            )
        except Exception:
            action_tier_rows = []
        self._last_action_salience_diagnostics = {
            "connected_component_salience_enabled": True,
            "salience_tiers_emitted": bool(tier_rows or action_tier_rows),
            "generation_stage_action_prioritization": hasattr(prior, "click_points"),
            "tier_rows": tier_rows,
            "action_tier_rows": action_tier_rows,
        }

    def action_salience_diagnostics(self) -> dict[str, Any]:
        """REQ-ARC-FCP-5397: expose live candidate-generation salience evidence."""

        return dict(self._last_action_salience_diagnostics)

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
            self._goal_bias_score_sum += score
            self._goal_bias_score_sumsq += score * score
            if self._goal_bias_score_min is None or score < self._goal_bias_score_min:
                self._goal_bias_score_min = score
            if self._goal_bias_score_max is None or score > self._goal_bias_score_max:
                self._goal_bias_score_max = score
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
            masked = g.copy()
            masked[self.hud_mask] = 0  # collapse counter/timer cells so equal game states dedup
            masked_hash = frame_hash(masked)
            guard = self._hud_collapse_guard
            if guard is not None and guard.is_split(masked_hash):
                # REQ-ARC-WMTE-5960 HARD REFUSAL. This masked hash was PROVEN to alias two
                # behaviourally distinct true states (same (node, concrete action) -> two
                # different masked successors, with the unmasked control showing only one). So
                # for this node -- or, past the guard's split cap, for every node -- identity
                # reverts to the unmasked frame. Appending the unmasked hash rather than
                # returning it bare keeps the compound key visibly derived from the masked one,
                # which makes a split node greppable in a dumped graph.
                return masked_hash + "|u:" + frame_hash(g)
            return masked_hash
        return frame_hash(g)

    def _unmasked_hash(self, frame_or_grid) -> Optional[str]:
        """Hash IGNORING `hud_mask` -- the collapse guard's mandatory control channel.

        Without this the guard could not distinguish "the mask collapsed two distinct states"
        from "this environment is simply non-deterministic at this node", and would fire
        spuriously (measured on sc25, where masked and unmasked violation counts match 1:1).
        """

        if frame_or_grid is None:
            return None
        try:
            from carnot.agentic.arc_agi3_world_model import grid_of, frame_hash

            return frame_hash(grid_of(frame_or_grid))
        except Exception:
            return None

    def hud_mask_diagnostics(self) -> dict:
        """Per-run HUD/identity evidence for an artifact row. Always safe to call."""

        summary = mask_summary(self.hud_mask)
        out = {
            "auto_hud_mask_enabled": bool(self.auto_hud_mask),
            "edge_bar_hud_mask_enabled": bool(self.edge_bar_hud_mask_enabled),
            "hud_mask_collapse_guard_enabled": bool(self.hud_mask_collapse_guard_enabled),
            "hud_mask_source": str(self._hud_mask_source),
            "hud_mask_resolved": bool(summary["resolved"]),
            "hud_mask_cell_count": int(summary["cell_count"]),
            "hud_mask_rows": summary["rows"],
            "hud_mask_cols": summary["cols"],
            "graph_nodes": int(len(self.graph)),
            "unique_frames": int(len(self._unique_frame_hashes)),
            # graph_nodes / unique UNMASKED frames. 1.0 means every distinct raw frame became
            # its own node (no dedup at all -- the r11l pathology); below 1.0 means the mask is
            # collapsing frames together. This is the LEGAL, oracle-free stand-in for true node
            # inflation, which would need a per-game count of real game states (available only
            # from the public game's source, and therefore diagnostic-only).
            "node_inflation_vs_unique_frames": (
                round(len(self.graph) / max(1, len(self._unique_frame_hashes)), 4)
                if self._unique_frame_hashes
                else None
            ),
        }
        if self._hud_collapse_guard is not None:
            out["collapse_guard"] = self._hud_collapse_guard.diagnostics()
            out["collapse_guard_refusals"] = int(self._hud_collapse_guard.refusals)
        else:
            out["collapse_guard"] = None
            out["collapse_guard_refusals"] = 0
        return out

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
        cps_diag: dict = {}
        candidates = rich_action_candidates(
            frame,
            frame_change_scorer=self.frame_change_scorer,
            frame_change_prune_threshold=self.frame_change_prune_threshold,
            action_prior=action_prior,
            action_prior_prune_quantile=self.action_prior_prune_quantile,
            candidate_router=self.candidate_router,
            previous_frame=previous_frame,
            click_pixel_sampling=self.click_pixel_sampling_enabled,
            click_pixel_samples_per_component=self.click_pixel_samples_per_component,
            click_pixel_rng=self._cps_rng,
            click_pixel_diagnostics_out=cps_diag,
        )
        if self.click_pixel_sampling_enabled:
            self._cps_rows_sampled += sum(1 for c in candidates if int(c.action_id) == 6)
            # Accumulate the GENERATION path's real activity. An empty dict here means
            # rich_action_candidates never reached the sampler at all (no click action in the
            # frame's vocabulary), which is a legitimate zero -- distinct from a sampler that
            # ran and changed nothing, which reports points_in > 0 with coords_changed == 0.
            self._cps_coords_changed += int(cps_diag.get("coordinates_changed") or 0)
            self._cps_gen_errors += int(cps_diag.get("errors") or 0)
            self._cps_points_in += int(cps_diag.get("points_in") or 0)
            self._cps_points_out += int(cps_diag.get("points_out") or 0)
            self._cps_unresolved += int(cps_diag.get("unresolved") or 0)
            self._cps_contested_centroid_points += int(
                cps_diag.get("contested_centroid_points") or 0
            )
        self._record_action_salience_diagnostics(frame, candidates, action_prior)
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
        if self.inert_click_pruner is not None and rows:
            try:
                rows = self.inert_click_pruner.rank_candidates(frame, rows)
            except Exception:
                pass
        if self.generic_causal_primitive is not None and rows:
            try:
                rows = self.generic_causal_primitive.rank_candidates(frame, rows)
            except Exception:
                pass
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
        rows = self._apply_controllable_novelty_order(frame, rows)
        if self.epistemic_ledger is not None and rows:
            try:
                rows = self.epistemic_ledger.rank_candidates(
                    frame,
                    rows,
                    runtime_receipts={
                        "source": "StepwiseExplorer._candidates",
                        "candidate_count": len(rows),
                    },
                )
            except Exception:
                pass
        if self.structured_evidence_memory is not None and rows:
            try:
                rows = self.structured_evidence_memory.rank_candidates(
                    frame,
                    rows,
                    provenance={
                        "source": "StepwiseExplorer._candidates",
                        "candidate_count": len(rows),
                    },
                )
            except Exception:
                pass
        # REQ-ARC-WMTE-5836: stamp the just-explore priority tier LAST, after every ranker has
        # run. Deliberate ordering: the tier decides WHETHER a candidate is admitted yet, and the
        # existing rankers keep deciding the order WITHIN an admitted tier -- so the barrier is
        # additive to (not a replacement for) the proven candidate ordering. A stamping failure
        # leaves rows un-stamped, which reads back as tier 0 = always eligible = today's behaviour
        # (fails OPEN, never stalls the search).
        if self._tier_active(frame) and rows:
            try:
                # include_cells is MANDATORY when the coordinates may be sampled member
                # pixels rather than centroids (REQ-ARC-WMTE-5950): the centroid-keyed map
                # misses a sampled pixel ~100% of the time, every row then reads back as
                # tier 0 = always eligible, and the barrier silently becomes a no-op on
                # exactly the click games it was measured to help.
                rows = annotate_frontier_tiers(
                    rows, frame, include_cells=self.click_pixel_sampling_enabled
                )
            except Exception:
                pass
        return rows

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
        # REQ-ARC-WMTE-5836: mirror into the reverse index the distance gradient walks. Kept in
        # lock-step with self.adj HERE (the single forward-edge write path) rather than rebuilt on
        # demand, so the two indices cannot drift. Only state-CHANGING edges reach this point (the
        # self-edge guard above), which is exactly the reference's "success == 1" edge set -- a
        # gradient built over no-op edges would promise routes that do not exist.
        self.radj.setdefault(next_hash, []).append((act, origin))

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
        if not self._hud_mask_attempted:
            # One attempt, on the first observed frame -- a static mask for the episode's
            # lifetime (never refreshed mid-search: changing the mask after nodes are already
            # hashed under the old one would corrupt node identity, per _hash's masking).
            self._hud_mask_attempted = True
            if self.auto_hud_mask and self.hud_mask is None:
                self.hud_mask = _compute_hud_mask_from_frame(
                    latest, edge_bar_detector=self.edge_bar_hud_mask_enabled
                )
                self._hud_mask_source = (
                    (
                        "edge_bar_detector_req5960"
                        if self.edge_bar_hud_mask_enabled
                        else "status_bar_classifier_req5583"
                    )
                    if self.hud_mask is not None
                    else "unresolved_no_bar_detected"
                )
        unmasked_now = self._unmasked_hash(latest)
        # The UNMASKED hash of the frame the agent was standing on when it issued the action
        # that produced `latest` -- i.e. the raw antecedent. Read from the previous _ingest, NOT
        # from the graph node's stored frame: the bare explorer only RETAINS node frames when
        # one of several optional components is enabled, so `awaiting["previous_frame"]` is None
        # on every transition of a bare run (measured: 1952 of 1952 on tu93). Sourcing the
        # control from there is what left the collapse guard's control channel dead.
        unmasked_before = self._last_unmasked_hash
        if unmasked_now:
            self._unique_frame_hashes.add(unmasked_now)
            self._last_unmasked_hash = unmasked_now
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
            # REQ-ARC-WMTE-5960 COLLAPSE GUARD. One realized (origin, concrete action) ->
            # successor triple, fed to the guard with its mandatory unmasked control. If this
            # key has now been seen producing two DIFFERENT masked successors while the
            # unmasked control shows only one, the mask is PROVEN to be collapsing distinct
            # states at `origin`, and `_hash` starts keying that node by masked+unmasked from
            # here on (or, past the split cap, keys every node that way). Every proof is
            # counted in `hud_mask_diagnostics()["collapse_guard"]`, so the guard's activity is
            # never invisible. Wrapped defensively for the same reason every sibling observe
            # hook in this block is: a guard that crashes must not take the search down.
            guard = self._hud_collapse_guard
            if guard is not None and self.hud_mask is not None:
                try:
                    guard.observe(
                        origin_masked=o.get("origin"),
                        origin_unmasked=unmasked_before,
                        # CONCRETE action, not the action NAME. Keying on the name alone lumps
                        # every distinct click together and manufactures spurious violations
                        # (measured: 13 false ones on r11l). The guard coerces the payload
                        # (which may be a dict of click coordinates) to a stable key itself.
                        action_key=(int(o["action"]), o.get("data")),
                        successor_masked=h,
                        successor_unmasked=unmasked_now,
                    )
                except Exception:
                    pass
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
            # Task 9 follow-on (2026-07-13): feed InertClickSigPruner the same realized
            # (before, label, after, leveled_up) transition every sibling online-learning
            # component above gets, so its tally actually accumulates from the search's own
            # live clicks (matching HazardMovePruner's own online-observation discipline) --
            # without this, rank_candidates would be wired but permanently a no-op (every
            # signature stays "unproven" forever).
            inert_click_pruner = getattr(self, "inert_click_pruner", None)
            if inert_click_pruner is not None and o.get("previous_frame") is not None:
                try:
                    inert_click_pruner.observe(
                        o.get("previous_frame") or o.get("grid"),
                        {"action": int(o["action"]), "data": o.get("data")},
                        latest,
                        leveled_up=bool(level_increased),
                    )
                except Exception:
                    pass
            generic_causal_primitive = getattr(self, "generic_causal_primitive", None)
            if generic_causal_primitive is not None and o.get("previous_frame") is not None:
                try:
                    generic_causal_primitive.observe_transition(
                        o.get("previous_frame") or o.get("grid"),
                        int(o["action"]),
                        o.get("data"),
                        latest,
                        leveled_up=bool(level_increased),
                    )
                except Exception:
                    pass
            structured_evidence_memory = getattr(self, "structured_evidence_memory", None)
            if structured_evidence_memory is not None and o.get("previous_frame") is not None:
                try:
                    structured_evidence_memory.observe_action_result(
                        o.get("previous_frame") or o.get("grid"),
                        int(o["action"]),
                        o.get("data"),
                        latest,
                        level_before=int(o.get("level_before") or 0),
                        level_after=int(lvl),
                        provenance={
                            "source": "StepwiseExplorer._ingest.after_transition",
                            "level_increased": bool(level_increased),
                        },
                    )
                except Exception:
                    pass
            cycle_allows_update = True
            cycle_verifier = getattr(self, "transition_cycle_verifier", None)
            if cycle_verifier is not None and o.get("previous_frame") is not None:
                cycle_allows_update = False
                try:
                    decision = cycle_verifier.observe_transition(
                        o.get("previous_frame") or o.get("grid"),
                        int(o["action"]),
                        o.get("data"),
                        latest,
                    )
                    if bool(getattr(decision, "admitted", False)):
                        cycle_allows_update = True
                        self._transition_cycle_admitted += 1
                        receipt = getattr(decision, "update_receipt", None)
                        if isinstance(receipt, dict):
                            self._transition_cycle_receipts.append(dict(receipt))
                    elif bool(getattr(decision, "abstained", False)):
                        self._transition_cycle_abstained += 1
                    else:
                        self._transition_cycle_rejected += 1
                except Exception:
                    self._transition_cycle_abstained += 1
            if level_increased:
                # REQ-ARC-WMTE-5836: reset the GLOBAL tier barrier on a level-up. The reference
                # goes further and discards its whole graph on level-up; Carnot deliberately keeps
                # its graph (cross-level navigation and the banked path are load-bearing here in a
                # way they are not there), so only the barrier is reset. Rationale: a new level is a
                # NEW board, and an active tier of 3 carried over from the previous level would
                # start the fresh board by clicking dull/large segments while its button-like
                # tier-0 objects sat untried -- inverting the mechanism's entire point.
                if self._tier_active() and self._active_tier != 0:
                    self._active_tier = 0
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
                if cycle_allows_update:
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
                                or self.click_pixel_sampling_enabled
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
                        or self.click_pixel_sampling_enabled
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
        # REQ-ARC-WMTE-5836: advance the GLOBAL tier barrier BEFORE computing eligibility, so a
        # milestone where the active tier just went empty everywhere admits the next tier in the
        # same decision instead of reporting a spurious "explored out". No-op when the flag is off.
        self._maybe_advance_tier()
        eligible: list[tuple[str, dict, int, float, float, float, float]] = []
        for h, node in self.graph.items():
            if not self._node_has_open_tier(node):
                # A node whose remaining work is merely TIER-DEFERRED is not exhausted -- do not
                # feed the online discriminator a false negative for it (see
                # _node_is_tier_deferred). Genuinely-empty nodes are recorded exactly as before.
                if not self._node_is_tier_deferred(node):
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

        # REQ-ARC-WMTE-5836 MECHANISM (b): prefer the navigation-NEAREST eligible node over the
        # shallowest-from-root one. Applied here, AFTER discriminative pruning and the tier gate,
        # so the gradient can only ever choose among nodes the existing machinery already deemed
        # expandable -- it reorders the ELIGIBLE SET, it never widens it. Falls through to the
        # historical key ordering whenever the gradient has no opinion (nothing reachable over
        # known-working edges). One thing it must ALSO not do, and did not do correctly until the
        # 2026-07-24 fix: it must not hand back a depth-capped self.cur, because the caller
        # expands a returned self.cur IN PLACE with no depth test, which would re-enable a branch
        # the max_depth cap had deliberately abandoned. See _gradient_frontier_target.
        if self.frontier_gradient_enabled and eligible:
            gradient_target = self._gradient_frontier_target([h for h, *_ in eligible])
            if gradient_target is not None:
                return gradient_target

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
        if self._tier_active():
            # REQ-ARC-WMTE-5836: batch only TIER-ADMITTED rows. Popping a deferred row here would
            # silently defeat the barrier (the batch is expanded unconditionally downstream), and
            # popping nothing at all would leave next_move with an empty pending queue -> the
            # _serve() IndexError. Callers therefore only reach here when _node_has_open_tier is
            # true, and this returns at least one row whenever that holds.
            eligible = self.tier_policy.eligible_indices(node["untested"], self._active_tier)
            count = min(int(limit), len(eligible))
            picked = eligible[:count]
            actions = [node["untested"][i] for i in picked]
            for i in sorted(picked, reverse=True):
                del node["untested"][i]
            return actions
        count = min(int(limit), len(node["untested"]))
        actions = node["untested"][:count]
        del node["untested"][:count]
        return actions

    def _pop_untested(self, node):
        """Pop the next untested action: the most-salient (pop(0)) normally; but when hybrid diversity is on
        AND the search has STALLED (no new level for _stall_threshold moves), pop a RANDOM one among the
        top-K -- the injection that recovers the structure-missed wins (r11l/sp80) the depth-first ride over-
        commits past. Flag OFF -> always pop(0) -> byte-identical to the submitted behavior.

        REQ-ARC-WMTE-5836: when the just-explore tier barrier is on, the pop is restricted to rows
        the GLOBAL active tier admits (and, on the uniform-random arm, drawn uniformly among them
        rather than greedily). This IS the live click decision the whole graft targets -- the
        coordinate-blind learned router leaves this line deciding the order by itself."""
        row = self._pop_untested_inner(node)
        if self.click_pixel_sampling_enabled:
            self._cps_maybe_redraw(node, row)
        return row

    def _cps_maybe_redraw(self, node: dict, row: Any) -> None:
        """REQ-ARC-WMTE-5950 -- the bounded WITH-REPLACEMENT half of the sampling rule.

        WHY A NEW ROW IS NEEDED AT ALL. ``node["untested"]`` is built exactly once per
        frame-hash and every other reference to it DELETES from it -- there is no refill or
        re-sample path anywhere in this class. So without this method a click object is
        tried at exactly one pixel, exactly once, forever, and the reference's
        redraw-on-revisit behaviour is not merely absent but structurally inexpressible.
        This is the MINIMAL change that permits revisiting: on popping a click row, resolve
        which object that coordinate belongs to and, while the per-(node, object) draw
        budget allows, append ONE more row for the same object at a fresh uniform pixel.

        HOW IT DIFFERS FROM THE REFERENCE, honestly. The reference retires an object only
        when a click on it produced NO state change, and otherwise redraws indefinitely.
        Attributing "did this change the frame" here would require hooking outcome
        attribution (which happens a step later, at a different frame), so this uses a
        fixed budget instead: at most ``click_pixel_redraw_budget`` draws per (node,
        object). Bounded by construction, so it cannot livelock a node -- which an
        outcome-keyed version could, since a working object stays selectable forever.

        Every decline path is COUNTED (budget / unresolved / no-frame / error). An
        uninstrumented mechanism is how this project previously read a 72-97% crashed arm
        as a legitimate null across 975 cells.
        """

        if self.click_pixel_redraw_budget <= 1 or not isinstance(row, Mapping):
            return
        try:
            if int(row.get("action")) != 6:
                return
            data = row.get("data")
            if not isinstance(data, Mapping):
                return
            x, y = int(data["x"]), int(data["y"])
        except Exception:
            self._cps_errors += 1
            return
        frame = node.get("frame")
        if frame is None:
            self._cps_redraws_declined_no_frame += 1
            return
        try:
            # component_partition accepts a live frame directly (it unwraps `.frame` and
            # takes the LAST sub-grid), so no grid coercion is needed here.
            #
            # KNOWN IMPRECISION, bounded and fail-safe: resolution checks the CENTROID key
            # before cell containment (so a generated centroid traces to the object that
            # produced it, which is the common case). If a sampled member pixel of object A
            # happens to also BE object B's truncated centroid, this attributes the redraw to
            # B -- the budget is charged to B and the next pixel is drawn from B. The result
            # is still a valid click on a real object and is still bounded by the budget, so
            # it costs a little ordering precision, never correctness or termination.
            partition = click_component_partition(frame)
            index, point = redraw_click_component_pixel(
                frame, x, y, rng=self._cps_rng, partition=partition
            )
        except Exception:
            self._cps_errors += 1
            return
        if index is None or point is None:
            self._cps_redraws_declined_unresolved += 1
            return
        ledger = node.setdefault("_cps_draws", {})
        drawn = int(ledger.get(index, 1))  # the row just popped IS draw #1
        if drawn >= self.click_pixel_redraw_budget:
            self._cps_redraws_declined_budget += 1
            return
        ledger[index] = drawn + 1
        # The tier is carried over unchanged rather than re-derived. It is invariant BY
        # CONSTRUCTION -- the new pixel belongs to the same object, so same colour and same
        # bounding box, so the same 5-tier predicate result -- and copying it keeps the
        # redraw from needing a tier map here. Asserted in the test suite.
        fresh: dict[str, Any] = {
            "action": 6,
            "data": {"x": int(point[0]), "y": int(point[1])},
        }
        if "tier" in row:
            fresh["tier"] = row["tier"]
        node["untested"].append(fresh)
        self._cps_redraws += 1

    def _pop_untested_inner(self, node):
        """The pop itself. Split out from ``_pop_untested`` so REQ-ARC-WMTE-5950's redraw hook
        wraps EVERY return path (tier-admitted draw, tier fail-open, stall-diversity draw, plain
        pop(0)) instead of having to be repeated at four ``return`` statements. Note the redraw
        deliberately does NOT wrap ``_pop_frontier_batch``: a batch is expanded wholesale
        downstream, so re-appending mid-batch would change batch semantics as well as the
        coordinate, and this experiment varies one thing at a time."""

        lst = node["untested"]
        if self._tier_active():
            rng = self._fd_rng if self.tier_uniform_random_enabled else None
            # top_k=None => UNRESTRICTED uniform draw over every tier-admitted row, which is what
            # the reference actually does (graph_explorer.choose_edge: `random.choice(untested_
            # edges)` accumulated over groups 0..active_group, with no top-k of any kind).
            # This used to read self._div_topk (default 8) -- the knob belonging to the UNRELATED
            # hybrid-diversity feature. That was wrong twice over: it silently made the arm a
            # top-8 draw instead of the reference's uniform-over-all draw (on r11l a node carries
            # ~34 candidates, so the distributions differ materially), and it coupled an A/B arm
            # to a foreign env var, so an operator tuning CARNOT_ARC_EXPLORE_DIV_TOPK for
            # diversity would have silently changed the experiment. The frontier-discipline draw
            # now has its OWN knob, unset by default = faithful to the reference.
            top_k = self._fd_draw_topk if self.tier_uniform_random_enabled else None
            idx = self.tier_policy.select_index(lst, self._active_tier, rng=rng, top_k=top_k)
            if idx is not None:
                return lst.pop(idx)
            # Nothing admitted at this tier. Defensive only: every caller gates on
            # _node_has_open_tier first, so reaching here means the barrier state and the node
            # disagree. Fail OPEN (take the row) rather than crash the live agent.
            self._tier_deferrals += 1
            return lst.pop(0)
        if (
            self._hybrid_diversity
            and self._steps_since_progress > self._stall_threshold
            and len(lst) > 1
        ):
            return lst.pop(self._div_rng.randrange(min(len(lst), self._div_topk)))
        return lst.pop(0)

    def _tier_active(self, frame: Any = None) -> bool:
        """Is the tier barrier active RIGHT NOW? (see SUBMITTED_FRONTIER_TIER_CLICK_VOCAB_ONLY_ENABLED)

        The barrier ranks CLICK-TARGET salience, so it is only defined on games that offer clicks.
        On a nav-only game it has no targets to rank and can only perturb a move ordering that was
        already working -- measured: arm B2 lost tu93 (the one nav-only baseline win) on 2 of 3 seeds
        in every condition of the full-spec A/B, while all of its gains were click games.

        Click availability is read from the frame the env hands us, via the same
        ``_available_action_ids`` helper the live adapter uses (enums / "ACTION6" / bare ints), and is
        LATCHED once seen -- a game that offers clicks on some frames but not others is still a click
        game. This is runtime discovery, legal on a game never seen before; it is NOT the harness's
        hardcoded CLICK_GAMES list.

        Fails OPEN toward today's behaviour: when the barrier is off, or when it is on and no click
        has been observed yet, this returns the value that leaves the search unmodified.
        """
        if not self.tier_exhaustion_enabled:
            return False
        if not self.tier_click_vocab_only:
            return True
        if frame is not None and not self._fd_click_vocab_seen:
            try:
                from carnot.agentic.arc_agi3_live_adapter import _available_action_ids

                if 6 in _available_action_ids(frame):
                    self._fd_click_vocab_seen = True
            except Exception:
                # Unparseable frame: do not latch, do not crash. Next frame gets another chance.
                pass
        return self._fd_click_vocab_seen

    def _node_has_open_tier(self, node: Mapping[str, Any] | None) -> bool:
        """Does this node still have untested work the GLOBAL barrier admits?

        With the barrier off this is exactly the historical ``bool(node["untested"])`` test, so
        every call site below stays byte-identical in the default configuration."""

        if not node:
            return False
        rows = node.get("untested") or []
        if not self._tier_active():
            return bool(rows)
        return self.tier_policy.node_has_open_tier(rows, self._active_tier)

    def _node_is_tier_deferred(self, node: Mapping[str, Any] | None) -> bool:
        """True when a node has untested work but ALL of it sits above the active tier.

        Needed to keep the online discriminative learner honest: ``_frontier`` records a
        NEGATIVE training sample ("frontier_exhausted") for every node it skips, and a
        merely-DEFERRED node is not exhausted -- it will be expanded later, once the barrier
        advances. Labelling it 0 now would teach the discriminator that a perfectly good node is
        a dead end, i.e. the barrier would silently poison a component that has nothing to do
        with it."""

        if not node:
            return False
        rows = node.get("untested") or []
        if not rows:
            return False
        return not self._node_has_open_tier(node)

    def _maybe_advance_tier(self) -> None:
        """Advance the GLOBAL priority barrier when NOTHING anywhere is still open at it.

        Delegates the decision to the pure ``TierExhaustionPolicy.next_active_tier`` (see that
        method's docstring for why global set-exhaustion is the faithful Carnot analogue of the
        reference's unreachability trigger, and why it may skip several tiers at once). This
        wrapper only owns the mutable state and the telemetry."""

        if not self._tier_active():
            return
        new_tier = self.tier_policy.next_active_tier(
            (node.get("untested") or [] for node in self.graph.values()),
            self._active_tier,
        )
        if new_tier != self._active_tier:
            self._tier_advances += new_tier - self._active_tier
            self._active_tier = new_tier

    def _cur_is_depth_capped(self) -> bool:
        """True when the depth-first ride has already been REFUSED for the current node.

        ``next_move`` step 1 rides the current node only while ``len(path) < max_depth``; past
        that cap the intent is explicitly to ABANDON this branch and go expand somewhere else.
        The frontier chooser must therefore know about the cap, because step 2's ``th == self.cur``
        branch expands in place with no depth test of its own."""

        node = self.graph.get(self.cur) if self.cur is not None else None
        if not node:
            return False
        try:
            return len(node.get("path") or []) >= int(self.max_depth)
        except Exception:
            return False

    def _gradient_frontier_target(self, eligible_hashes: Sequence[str]) -> Optional[str]:
        """MECHANISM (b): the navigation-NEAREST eligible frontier node, or None (no opinion).

        One multi-source reverse BFS seeded at every eligible node labels the whole known graph
        with hops-to-nearest-open-node; following the next-hop chain from ``self.cur`` names the
        node the gradient leads to. Returning None (nothing open, or no known-working route from
        here) makes the caller fall back to its existing depth-primary ordering -- the gradient is
        a PREFERENCE, never a veto, so it can never render a reachable node unreachable.

        DEPTH-CAP INTERACTION (fixed 2026-07-24 after an adversarial review; this is the
        load-bearing subtlety of the whole mechanism). ``self.cur`` is normally one of the
        eligible nodes, and a seed of the multi-source BFS sits at distance 0 -- so
        ``nearest_open_node`` would return ``self.cur`` itself EVERY time the current node still
        has open work, and the caller's ``th == self.cur`` branch would then expand it in place
        with no depth check. On deep-graph games that is precisely the state the caller is in
        when it reaches here: step 1 was skipped BECAUSE ``len(path) >= max_depth``. Measured on
        r11l before the fix: 140 of 140 gradient picks (and 195 of 195 at budget 800) were
        exactly this case, so what looked like "the distance gradient fired" was in fact "the
        max_depth=45 backtrack cap was silently cancelled" -- a completely different
        intervention, and one that confounded arms C/D beyond interpretation.

        The fix is to refuse ``self.cur`` as a GRADIENT SEED when it is at/over the depth cap.
        The gradient then does what it says: it chooses among the OTHER open nodes by
        navigation distance. When cur is under the cap it stays an eligible seed, because in
        ``best_first`` mode step 1 never runs and expanding cur in place is legitimate there.
        A cur that is depth-capped and is the ONLY open node yields no seeds -> None -> the
        caller falls back to its historical ordering, i.e. still never a veto."""

        if not self.frontier_gradient_enabled or self.cur is None or not eligible_hashes:
            return None
        seeds = list(eligible_hashes)
        cur_at_cap = self._cur_is_depth_capped()
        if cur_at_cap and self.cur in seeds:
            seeds = [h for h in seeds if h != self.cur]
            self._gradient_cur_at_cap_excluded += 1
        if not seeds:
            self._gradient_misses += 1
            return None
        try:
            field_ = frontier_distance_field(self.radj, seeds)
            target = nearest_open_node(field_, self.cur)
        except Exception:
            return None
        if target is None or target not in self.graph:
            self._gradient_misses += 1
            return None
        # Belt-and-braces: even with cur refused as a seed, never hand the caller a target that
        # equals a depth-capped cur (a malformed field could in principle walk back to it).
        if target == self.cur and cur_at_cap:
            self._gradient_misses += 1
            return None
        self._gradient_targets += 1
        if target == self.cur:
            self._gradient_pick_cur += 1
        else:
            self._gradient_pick_other += 1
        return str(target)

    def frontier_discipline_diagnostics(self) -> dict[str, Any]:
        """A/B telemetry for the two just-explore frontier-discipline mechanisms."""

        return {
            "tier_exhaustion_enabled": bool(self.tier_exhaustion_enabled),
            "tier_uniform_random_enabled": bool(self.tier_uniform_random_enabled),
            "tier_click_vocab_only": bool(self.tier_click_vocab_only),
            "tier_click_vocab_seen": bool(self._fd_click_vocab_seen),
            "tier_active_effective": bool(self._tier_active()),
            "frontier_gradient_enabled": bool(self.frontier_gradient_enabled),
            "tier_count": int(self.tier_count),
            "active_tier": int(self._active_tier),
            "tier_advances": int(self._tier_advances),
            "tier_deferral_fallbacks": int(self._tier_deferrals),
            # None = the reference's unrestricted uniform draw over all tier-admitted rows.
            "tier_draw_top_k": self._fd_draw_topk,
            "gradient_targets_chosen": int(self._gradient_targets),
            "gradient_misses": int(self._gradient_misses),
            # Pick-kind split (see _gradient_frontier_target). gradient_pick_other is the ONLY
            # kind that is the mechanism-under-test; gradient_pick_cur is an in-place expansion
            # of a node still under the depth cap; gradient_cur_at_cap_excluded counts the
            # decisions where a depth-capped current node was refused as a seed (which, before
            # the 2026-07-24 fix, were instead reported as successful gradient picks and
            # silently cancelled the max_depth backtrack cap).
            "gradient_pick_other_node": int(self._gradient_pick_other),
            "gradient_pick_current_node": int(self._gradient_pick_cur),
            "gradient_cur_at_depth_cap_excluded": int(self._gradient_cur_at_cap_excluded),
            "max_depth": int(self.max_depth),
            "reverse_edges": int(sum(len(v) for v in self.radj.values())),
            # REQ-ARC-WMTE-5950 -- per-object click-pixel sampling. Every counter is emitted
            # unconditionally (0 when the flag is off) so an arm can never be silently
            # uninstrumented: a reader can always tell whether the mechanism did anything.
            "click_pixel_sampling_enabled": bool(self.click_pixel_sampling_enabled),
            "click_pixel_samples_per_component": int(self.click_pixel_samples_per_component),
            "click_pixel_redraw_budget": int(self.click_pixel_redraw_budget),
            # NOT an activity counter: click rows PRESENT while the flag was on, which is the
            # same number for a working sampler and a dead one. Kept (renaming a shipped field
            # would break readers) but no longer the field anything reads for "did it fire".
            "click_pixel_rows_sampled": int(self._cps_rows_sampled),
            "click_pixel_rows_sampled_is_not_an_activity_counter": True,
            # THE ACTIVITY WITNESS. coordinates_changed > 0 is the only field that proves the
            # generation rule actually replaced a coordinate. A one-shot arm
            # (redraw_budget=1) issues zero redraws by design, so without this field it has
            # no evidence at all that the mechanism fired.
            "click_pixel_coordinates_changed": int(self._cps_coords_changed),
            "click_pixel_generation_errors": int(self._cps_gen_errors),
            "click_pixel_points_in": int(self._cps_points_in),
            "click_pixel_points_out": int(self._cps_points_out),
            "click_pixel_unresolved": int(self._cps_unresolved),
            "click_pixel_contested_centroid_points": int(self._cps_contested_centroid_points),
            "click_pixel_redraws": int(self._cps_redraws),
            "click_pixel_redraws_declined_budget": int(self._cps_redraws_declined_budget),
            "click_pixel_redraws_declined_unresolved": int(self._cps_redraws_declined_unresolved),
            "click_pixel_redraws_declined_no_frame": int(self._cps_redraws_declined_no_frame),
            # REDRAW-half errors only. Generation-path errors are
            # `click_pixel_generation_errors` above; summing the two is the mechanism's total.
            "click_pixel_errors": int(self._cps_errors),
        }

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
            # REQ-ARC-WMTE-5836: "exhausted" must mean "exhausted AT THE ACTIVE TIER" when the
            # barrier is on, or the go-explore replay would fire for a node that still has
            # perfectly good deferred work. Identical to the historical test when the flag is off.
            and (not self._node_has_open_tier(cur_node) or len(cur_node["path"]) >= self.max_depth)
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
            and self._node_has_open_tier(cur_node)
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
        if th is None and self._tier_active():
            # REQ-ARC-WMTE-5836: with the barrier on, "no eligible frontier" would be catastrophic
            # if it were ever reported while lower-priority work remained -- the run would end
            # early and the A/B would read as a null for a purely mechanical reason rather than a
            # behavioural one. _frontier already advances the barrier on entry (so the common case
            # is handled there, in the same decision), but the check is cheap and idempotent, so it
            # is repeated here as a belt-and-braces guard against any future call path that reaches
            # explored_out without having gone through _frontier's advance. A genuine explored-out
            # (nothing open even at the last tier) returns None from both calls.
            self._maybe_advance_tier()
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
        auto_hud_mask: bool = SUBMITTED_AUTO_HUD_MASK_ENABLED,
        edge_bar_hud_mask: bool | None = None,
        hud_mask_collapse_guard: bool | None = None,
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
        tier_exhaustion: bool | None = None,
        tier_count: int = SUBMITTED_FRONTIER_TIER_COUNT,
        tier_uniform_random: bool | None = None,
        tier_click_vocab_only: bool | None = None,
        frontier_gradient: bool | None = None,
        frontier_discipline_seed: int = 20260724,
        click_pixel_sampling: bool | None = None,
        click_pixel_samples_per_component: int = SUBMITTED_CLICK_PIXEL_SAMPLES_PER_COMPONENT,
        click_pixel_redraw_budget: int = SUBMITTED_CLICK_PIXEL_REDRAW_BUDGET,
        click_pixel_sampling_seed: int | None = None,
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
                auto_hud_mask=auto_hud_mask,
                edge_bar_hud_mask=edge_bar_hud_mask,
                hud_mask_collapse_guard=hud_mask_collapse_guard,
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
                tier_exhaustion=tier_exhaustion,
                tier_count=tier_count,
                tier_uniform_random=tier_uniform_random,
                tier_click_vocab_only=tier_click_vocab_only,
                frontier_gradient=frontier_gradient,
                frontier_discipline_seed=frontier_discipline_seed,
                click_pixel_sampling=click_pixel_sampling,
                click_pixel_samples_per_component=click_pixel_samples_per_component,
                click_pixel_redraw_budget=click_pixel_redraw_budget,
                click_pixel_sampling_seed=click_pixel_sampling_seed,
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
        auto_hud_mask: bool = SUBMITTED_AUTO_HUD_MASK_ENABLED,
        edge_bar_hud_mask: bool | None = None,
        hud_mask_collapse_guard: bool | None = None,
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
        tier_exhaustion: bool | None = None,
        tier_count: int = SUBMITTED_FRONTIER_TIER_COUNT,
        tier_uniform_random: bool | None = None,
        tier_click_vocab_only: bool | None = None,
        frontier_gradient: bool | None = None,
        frontier_discipline_seed: int = 20260724,
        click_pixel_sampling: bool | None = None,
        click_pixel_samples_per_component: int = SUBMITTED_CLICK_PIXEL_SAMPLES_PER_COMPONENT,
        click_pixel_redraw_budget: int = SUBMITTED_CLICK_PIXEL_REDRAW_BUDGET,
        click_pixel_sampling_seed: int | None = None,
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
        inert_click_pruner: Any | bool | None = SUBMITTED_INERT_CLICK_PRUNER_ENABLED,
        object_history_salience: Any | bool | None = SUBMITTED_OBJECT_HISTORY_SALIENCE_ENABLED,
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
        transition_cycle_verifier: Any | None = None,
        generic_causal_primitive: Any | None = None,
        epistemic_ledger: Any = _DEFAULT_EPISTEMIC_LEDGER,
        structured_evidence_memory: Any = _DEFAULT_STRUCTURED_EVIDENCE_MEMORY,
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
            candidate_router = _load_submitted_candidate_router(game_id=self.short)
        if frame_change_scorer is _DEFAULT_FRAME_CHANGE_SCORER:
            frame_change_scorer = _load_submitted_frame_change_scorer()
        if action_prior is None and SUBMITTED_COLOR_BLOB_SALIENCE_ENABLED:
            action_prior = ColorBlobSaliencePrior()
        action_prior = coerce_object_history_salience_prior(
            object_history_salience, base_prior=action_prior
        )
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
        if epistemic_ledger is _DEFAULT_EPISTEMIC_LEDGER:
            epistemic_ledger = SUBMITTED_EPISTEMIC_LEDGER_ENABLED
        self.epistemic_ledger = coerce_epistemic_ledger(epistemic_ledger)
        if structured_evidence_memory is _DEFAULT_STRUCTURED_EVIDENCE_MEMORY:
            structured_evidence_memory = SUBMITTED_STRUCTURED_EVIDENCE_MEMORY_ENABLED
        self.structured_evidence_memory = coerce_structured_evidence_memory(
            structured_evidence_memory
        )
        self.approach_recommendation["strategy"] = self.strategy_route
        self._route_from_frame_checked = False
        self._feature_router_checked = False
        self.feature_router: dict[str, Any] | None = None
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
            auto_hud_mask=auto_hud_mask,
            edge_bar_hud_mask=edge_bar_hud_mask,
            hud_mask_collapse_guard=hud_mask_collapse_guard,
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
            tier_exhaustion=tier_exhaustion,
            tier_count=tier_count,
            tier_uniform_random=tier_uniform_random,
            tier_click_vocab_only=tier_click_vocab_only,
            frontier_gradient=frontier_gradient,
            frontier_discipline_seed=frontier_discipline_seed,
            click_pixel_sampling=click_pixel_sampling,
            click_pixel_samples_per_component=click_pixel_samples_per_component,
            click_pixel_redraw_budget=click_pixel_redraw_budget,
            click_pixel_sampling_seed=click_pixel_sampling_seed,
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
            inert_click_pruner=inert_click_pruner,
            amortized_first_contact_prior=amortized_first_contact_prior,
            go_explore_archive=go_explore_archive,
            similarity_retrieval=similarity_retrieval,
            generic_causal_primitive=generic_causal_primitive,
            epistemic_ledger=self.epistemic_ledger,
            structured_evidence_memory=self.structured_evidence_memory,
        )
        self.transitions: list = []  # (grid_before, action, data, grid_after) self-collected
        self.explore_budget = (
            int(explore_budget)
            if explore_budget is not None
            else _route_explore_budget(self.strategy_route)
        )
        self.proposer = proposer  # default set lazily to LocalGGUFProposer
        self.transition_cycle_verifier = transition_cycle_verifier
        self.phase = "explore"
        self.plan: list = []
        self.pi = 0
        self._prev = None  # last (grid, action_id, data) for transition pairing
        self._prev_level = 0  # real level AT THE TIME self._prev was captured (see next_move)
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

    def _maybe_route_from_transitions(self) -> None:
        """REQ-CAPSTONE-4582 (live wiring): once enough of THIS game's own transitions have been
        observed, classify their behavioral signature (avatar motion / click-connect / config-toggle
        / hidden-carry-state / keyboard-vs-click effect density -- see
        `arc_solve_learning.extract_early_play_signature`) and use it to bias search the same way
        `_maybe_route_from_frame`'s frame-only hint already does: by adjusting whether the
        goal-distance heuristic portfolio applies and re-deriving the explore budget from it.

        This infers a KIND of game from how it has behaved so far in THIS live session -- never a
        game-identity lookup, never the executable win-check, never a per-game answer -- and is a
        strict prioritization signal over the explorer's own existing candidate generation, not a
        replacement for it. Fires once per game, guarded by `_feature_router_checked`.

        HONEST PRIOR RESULT (do not silently contradict it): `results/experiment_4582_feature_router_
        transfer.json` measured the CLOSELY-RELATED idea of switching the entire offline solver
        approach by classified mechanic and found NO value on the public held-out-variant methodology
        (transfer_delta=0.0, did not beat a random-route positive control; chosen_submitted_config:
        unchanged). What this method wires is narrower -- it never swaps solvers, it only adjusts one
        existing budget-sizing flag inside the SAME unified E3AgentPolicy cascade -- but it inherits
        the same classifier, so it is gated on `_FEATURE_ROUTER_MIN_CONFIDENCE` and treated as
        provisional/observational (always populates `self.feature_router` for visibility) rather than
        trusted to move the scored win rate until it has its own controlled measurement.

        RUN-LOCAL CROSS-GAME EXTENSION (opt-in via CARNOT_ARC_RUN_LOCAL_ADAPTATION=1, off by
        default; scoped in docs/research-notes/arc-agi3-run-local-cross-game-adaptation-scope-
        2026-07-12.md, operator-cleared to build): when enabled, this method also contributes this
        game's (mechanic_class, approach) choice to `_RUN_LOCAL_LEDGER` -- a thread-safe, in-process,
        NEVER-persisted-to-disk ledger shared by every game running CONCURRENTLY in this same
        submission (Swarm.main() runs the whole roster on separate threads in one process; confirmed
        against the real scored submission path, not just the offline reference) -- and consults that
        ledger for a small, bounded confidence bonus on top of the per-game classification before the
        SAME `_FEATURE_ROUTER_MIN_CONFIDENCE` gate above. This is learning a MECHANIC CLASS's behavior
        from this run's own concurrent play, never a memorized per-game action sequence (the
        operator's dividing line, recorded in the scope doc) and never a lookup into
        ops/arc_solve_registry.yaml. Still just a prototype: the scope doc's SS2.4 concurrent-play
        offline A/B validation has not run yet, so this stays off by default until it does."""

        if self._feature_router_checked:
            return
        if len(self.transitions) < _FEATURE_ROUTER_EARLY_PLAY_K:
            return
        self._feature_router_checked = True
        try:
            rows = _early_play_rows(self.transitions)
            recommendation = _recommend_live_approach(self.short, early_play_signature=rows)
        except Exception:
            return
        feature_router = (
            recommendation.get("feature_router") if isinstance(recommendation, dict) else None
        )
        if not isinstance(feature_router, dict) or not feature_router.get("enabled"):
            return
        self.feature_router = feature_router
        mechanic_class = str(feature_router.get("mechanic_class") or "")
        approach = str(feature_router.get("approach") or "")
        confidence = float(feature_router.get("confidence") or 0.0)
        if not mechanic_class or mechanic_class == "unknown" or not approach:
            return
        if _run_local_adaptation_enabled():
            # Contribute this game's own (mechanic_class, approach) choice to the run-local
            # ledger for OTHER concurrently-running games in this same submission run to
            # consult -- regardless of whether THIS game's own confidence gate ends up accepting
            # the nudge below. `id(self)` is a stable per-instance key for this process's
            # lifetime (see RunLocalMechanicLedger's docstring for the overwrite semantics).
            _RUN_LOCAL_LEDGER.update(
                mechanic_class, id(self), approach, _run_local_outcome_proxy(self.transitions)
            )
            confidence += _run_local_confidence_bonus(mechanic_class, approach)
        if confidence < _FEATURE_ROUTER_MIN_CONFIDENCE:
            # Classified, stored on self.feature_router for observability, but not confident
            # enough to change live search behavior -- see the confidence-gate rationale above.
            return
        # The behavioral 7-class taxonomy (avatar_navigation / click_connect / config_toggle /
        # hidden_carry_state / keyboard_graph / click_graph) is deliberately NOT the same vocabulary
        # as arc_strategy_router's 5 STRATEGY_CLASSES (program_editor / graph_explore / ...) -- an
        # unrecognized mechanic name there silently no-ops to the graph_explore default. The one
        # concrete, already-load-bearing knob both sides share is `uses_goal_distance_heuristic`
        # (it gates the explore-budget portfolio in `_route_explore_budget`); only
        # `goal_distance_astar` genuinely implies avatar/goal-distance structure is present.
        wants_goal_distance = approach == "goal_distance_astar"
        if bool(self.strategy_route.get("uses_goal_distance_heuristic")) != wants_goal_distance:
            routed = dict(self.strategy_route)
            routed["uses_goal_distance_heuristic"] = wants_goal_distance
            routed["feature_router_mechanic_class"] = mechanic_class
            routed["feature_router_approach"] = approach
            routed["feature_router_confidence_with_run_local_bonus"] = confidence
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

    def _novelty_observed_stack(self):
        """REQ-ARC-FCP-5699-19: stack of every real (before, after) grid observed this episode,
        for the novelty goal-energy fallback used on first-contact levels -- games that have never
        completed a level yet have no ``_previous_level_complete_grid`` exemplar (REQ-ARC-FCP-
        5699-18: that makes ``_goal_energy_for_plan``'s graded branch structurally inapplicable,
        collapsing the search's guidance to a flat, zero-gradient constant). Grids with a shape
        different from the first grid's are dropped (defensive; real episodes should be
        shape-consistent). Returns None if no grids are available."""
        import numpy as np

        try:
            active = self._active_transitions()
        except Exception:
            return None
        grids = []
        for t in active:
            try:
                grids.append(np.asarray(t.grid))
                grids.append(np.asarray(t.next_grid))
            except Exception:
                continue
        if not grids:
            return None
        shape0 = grids[0].shape
        grids = [g for g in grids if g.shape == shape0]
        if not grids:
            return None
        return np.stack(grids, axis=0)

    def _goal_energy_for_plan(self, is_done):
        """SCENARIO-ARC-WMTE-4821-LIVE-PLAN-WIRING: lambda-gated model planner energy.

        REQ-ARC-FCP-5699-19 adds a NOVELTY fallback (DEV-ONLY, opt-in via
        ``CARNOT_ARC_NOVELTY_GOAL_BIAS=1``, unset in production) for the exemplar-free case
        REQ-ARC-FCP-5699-18 root-caused: when no ``_previous_level_complete_grid`` exists yet
        (a first-contact, never-completed level), the graded-exemplar branch cannot apply, and the
        function used to fall back to a flat constant that provides the best-first search with
        ZERO gradient. The novelty fallback instead scores each candidate grid by its distance to
        the NEAREST grid already concretely observed in the real episode (execution-grounded, no
        exemplar needed): states far from anything already seen get LOW energy (attractive to the
        min-heap search), states identical to something already seen get the same flat energy as
        before (no worse than the pre-existing binary fallback). This is a go-explore-flavored
        proxy for "unexplored territory is more likely to contain progress" -- it does not target
        the actual (unknown) goal directly, unlike the graded-exemplar branch, so it is a weaker
        signal in principle; it is opt-in pending empirical A/B validation, not a default flip.
        The returned closure's ``energy_source`` attribute records which branch fired
        (``"graded_exemplar"`` / ``"novelty"`` / ``"binary"``) for diagnostics."""

        if self.goal_guidance_lambda <= 0.0 or not callable(is_done):
            return None

        import os

        import numpy as np

        exemplar = getattr(self, "_previous_level_complete_grid", None)
        use_graded = os.environ.get("CARNOT_ARC_GRADED_GOAL_BIAS") == "1" and exemplar is not None
        exemplar_arr = np.asarray(exemplar) if use_graded else None
        scale = float(self.goal_guidance_lambda)

        observed_stack = None
        if not use_graded and os.environ.get("CARNOT_ARC_NOVELTY_GOAL_BIAS") == "1":
            observed_stack = self._novelty_observed_stack()

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
            if observed_stack is not None:
                g = np.asarray(grid)
                if g.shape != observed_stack.shape[1:]:
                    return scale
                diffs = np.mean(observed_stack != g[np.newaxis, ...], axis=(1, 2))
                min_diff_to_observed = float(diffs.min())
                return scale * (1.0 - min_diff_to_observed)
            return scale

        _energy.energy_source = (
            "graded_exemplar"
            if exemplar_arr is not None
            else "novelty"
            if observed_stack is not None
            else "binary"
        )
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

    @staticmethod
    def _planner_accepts_diagnostics(plan_in_model) -> bool:
        import inspect

        try:
            signature = inspect.signature(plan_in_model)
        except (TypeError, ValueError):
            return True
        if "diagnostics" in signature.parameters:
            return True
        return any(
            param.kind is inspect.Parameter.VAR_KEYWORD for param in signature.parameters.values()
        )

    @staticmethod
    def _planner_accepts_max_nodes(plan_in_model) -> bool:
        import inspect

        try:
            signature = inspect.signature(plan_in_model)
        except (TypeError, ValueError):
            return True
        if "max_nodes" in signature.parameters:
            return True
        return any(
            param.kind is inspect.Parameter.VAR_KEYWORD for param in signature.parameters.values()
        )

    def _call_plan_in_model(
        self,
        plan_in_model,
        engine,
        is_done,
        start_grid,
        *,
        diagnostics=None,
        goal_energy_override=None,
    ):
        import os

        # goal_energy_override (REQ-ARC-WMTE-5845): a caller-supplied, model-specific energy (e.g. the
        # structured nav model's player->goal Manhattan) takes precedence over the generic exemplar/novelty
        # derivation -- a strong nav-specific gradient makes plan_in_model best-first + more robust (fewer
        # nodes -> finds the plan within budget on bigger mazes). None (all other paths) preserves the
        # existing auto-derivation exactly.
        goal_energy = (
            goal_energy_override
            if goal_energy_override is not None
            else self._goal_energy_for_plan(is_done)
        )
        kwargs: dict = {}
        if goal_energy is not None and self._planner_accepts_goal_energy(plan_in_model):
            kwargs["goal_energy"] = goal_energy
        if diagnostics is not None:
            diagnostics["goal_energy_source"] = getattr(goal_energy, "energy_source", None)
        if diagnostics is not None and self._planner_accepts_diagnostics(plan_in_model):
            kwargs["diagnostics"] = diagnostics
        # DEV-ONLY diagnostic override (REQ-ARC-FCP-5699-15 follow-up): unset in production, so
        # this changes nothing by default. Lets an A/B/diagnostic script raise plan_in_model's
        # search budget past its 20000-node default without editing production call sites.
        _max_nodes_override = os.environ.get("CARNOT_ARC_PLAN_MAX_NODES")
        if _max_nodes_override and self._planner_accepts_max_nodes(plan_in_model):
            kwargs["max_nodes"] = int(_max_nodes_override)
        return plan_in_model(engine, is_done, start_grid, **kwargs)

    def _guided_plan_in_model(self, plan_in_model):
        def _wrapped(engine, is_done, start_grid):
            return self._call_plan_in_model(plan_in_model, engine, is_done, start_grid)

        return _wrapped

    def _next_plan_move(self) -> tuple:
        step = self.plan[self.pi]
        self.pi += 1
        move = (step["action"], step.get("data"))
        if self.structured_evidence_memory is not None:
            try:
                self.structured_evidence_memory.observe_action_candidate(
                    int(move[0]),
                    move[1],
                    provenance={
                        "source": "E3AgentPolicy._next_plan_move",
                        "phase": self.phase,
                    },
                )
            except Exception:
                pass
        return move

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
            self._prev_level = _level_of(latest)
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
            # Graduated to default 4096/600 (REQ-ARC-FCP-5699-35, was 2560/300 dev-only override
            # per REQ-ARC-FCP-5699-28): live evidence (a g50t refactor-loop round genuinely hit
            # "[HIT n_predict=2560 OUTPUT LIMIT before completing]") confirmed max_tokens was a
            # real bottleneck for the induce/refactor calls this proposer serves; the sge arm's
            # separate 300s TimeoutError showed the old timeout was already sometimes
            # insufficient at the old budget. REQ-ARC-FCP-5699-32 validated 4096/600 reaches
            # proposer_ok=True on 6/6 rounds of a real full-budget live run for THIS (9B) model --
            # the 8192 requirement found in REQ-ARC-FCP-5699-34 was specific to a 3x larger,
            # non-live candidate model, not this one. Both env vars remain overridable.
            self.proposer = LocalGGUFProposer(
                repo_substr="Qwen3.5-9B-MTP",
                model_path=os.environ.get("CARNOT_ARC_GGUF_PATH") or None,
                mtp=(os.environ.get("CARNOT_ARC_MTP", "1") != "0"),
                kv_quant="q8_0",
                no_think_prefix="/no_think\n",
                max_tokens=int(os.environ.get("CARNOT_ARC_INDUCE_MAX_TOKENS", "4096")),
                timeout=int(os.environ.get("CARNOT_ARC_INDUCE_TIMEOUT", "600")),
                port=int(os.environ.get("CARNOT_ARC_PROPOSER_PORT", "8919")),
                n_gpu_layers=int(os.environ.get("CARNOT_ARC_NGL", "999")),
            )
        return self.proposer

    def _embed_playbook_query(self, text: str):
        """REQ-ARC-WMTE-5718: embed the stuck-situation query with the SAME GGUF weights the
        proposer uses (live_llm_embedding_extraction), so the vector matches the offline index
        space. Lazily creates + caches an embedding-mode llama_cpp.Llama; fully guarded so any
        failure (missing GGUF, OOM) returns None and the caller falls back -- never crashes the
        live path. NOTE: this is a second in-memory load of the same weights; on a tight live GPU
        it must be VRAM-budgeted (the feature is dev-gated OFF, so the frozen submit is unaffected)."""
        import os

        embedder = getattr(self, "_playbook_embedder", None)
        if embedder is False:  # a prior attempt failed; do not retry every stall
            return None
        try:
            if embedder is None:
                from llama_cpp import Llama
                from llama_cpp.llama_cpp import LLAMA_POOLING_TYPE_LAST

                from carnot.agentic.arc_executable_world_model import _resolve_gguf

                gguf = os.environ.get("CARNOT_ARC_GGUF_PATH") or _resolve_gguf("Qwen3.5-9B")
                if not gguf:
                    self._playbook_embedder = False
                    return None
                embedder = Llama(
                    model_path=gguf,
                    embedding=True,
                    pooling_type=LLAMA_POOLING_TYPE_LAST,
                    n_ctx=2048,
                    n_gpu_layers=int(os.environ.get("CARNOT_ARC_NGL", "999")),
                    verbose=False,
                )
                self._playbook_embedder = embedder
            import numpy as np

            raw = embedder.embed(text, normalize=False, truncate=True)
            arr = np.asarray(raw, dtype=np.float32)
            return arr if arr.ndim == 1 else arr.reshape(-1)
        except Exception:
            self._playbook_embedder = False
            return None

    def _playbook_query_text(self, active_transitions) -> str:
        """REQ-ARC-WMTE-5718: a compact free-text description of the current stuck situation to
        embed as the retrieval query -- game id + grid shape + the action types and any click/
        no-op signals observed so far. Deliberately game-agnostic (no per-game facts): it is the
        SHAPE of the stuck situation, so it also works for a hidden game."""
        actions: list[int] = []
        shape = ""
        for t in active_transitions or ():
            try:
                actions.append(int(getattr(t, "action", 0)))
                if not shape:
                    shape = "x".join(str(d) for d in getattr(t, "grid").shape)
            except Exception:
                continue
        uniq = sorted(set(actions))
        has_click = 6 in uniq
        return (
            f"ARC-AGI-3 game {self.short}: the agent is stuck making no level progress on a "
            f"{shape or 'grid'} board; observed action types {uniq}"
            f"{'; uses click/coordinate actions' if has_click else '; keyboard/directional actions'}; "
            f"needs an exploration strategy to induce the world model and find the win condition."
        )

    def _retrieve_playbook_block(self, active_transitions):
        """REQ-ARC-WMTE-5718: retrieve the top-K playbook patterns for the current stuck situation
        and format them for injection. Returns the block string, or None on any failure (index
        missing, embedder unavailable) so the caller falls back to the static block or nothing."""
        try:
            from carnot.agentic import arc_playbook_retrieval as rag

            index = getattr(self, "_playbook_index", None)
            if index is None:
                index = self._playbook_index = rag.load_index()
            vec = self._embed_playbook_query(self._playbook_query_text(active_transitions))
            if vec is None:
                return None
            tags = rag.infer_query_mechanic_tags(game=self.short)
            top = rag.retrieve(index, vec, top_k=PLAYBOOK_RETRIEVAL_TOPK, query_tags=tags)
            block = rag.format_injection(top)
            return block or None
        except Exception:
            return None

    def _world_model_candidates(self, engine, is_done) -> list[WorldModelCandidate]:
        import os

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

        if self.epistemic_ledger is not None and latest is not None:
            try:
                self.epistemic_ledger.observe_state(
                    latest,
                    runtime_receipts={
                        "source": "E3AgentPolicy.next_move.before_routing",
                        "phase": self.phase,
                        "frames_seen": len(frames),
                    },
                )
            except Exception:
                pass
        if self.structured_evidence_memory is not None and latest is not None:
            try:
                self.structured_evidence_memory.observe_state(
                    latest,
                    phase=self.phase,
                    provenance={
                        "source": "E3AgentPolicy.next_move.before_routing",
                        "frames_seen": len(frames),
                    },
                )
            except Exception:
                pass
        self._maybe_route_from_frame(latest)
        # collect a transition from the last action's outcome
        if self._prev is not None and latest is not None:
            from carnot.agentic.arc_agi3_world_model import grid_of
            from carnot.agentic.arc_executable_world_model import Transition

            try:
                g0, aid, data = self._prev
                g1 = to_logical(grid_of(latest), self.cell)
                transition = Transition(g0, aid, data, g1, self._prev_level, _level_of(latest))
                admit_world_model_update = True
                verifier = getattr(self, "transition_cycle_verifier", None)
                if verifier is not None:
                    admit_world_model_update = False
                    try:
                        decision = verifier.observe_transition(g0, int(aid), data, g1)
                        admit_world_model_update = bool(getattr(decision, "admitted", False))
                    except Exception:
                        admit_world_model_update = False
                if admit_world_model_update:
                    self.transitions.append(transition)
                    self._dsl_transitions.append((g0, _action_key(aid, data), g1))
                    if self.epistemic_ledger is not None:
                        try:
                            self.epistemic_ledger.observe_transition(
                                self._prev[0],
                                int(aid),
                                data,
                                g1,
                                level_before=int(self._prev_level),
                                level_after=int(_level_of(latest)),
                                runtime_receipts={
                                    "source": "E3AgentPolicy.next_move.after_transition",
                                    "admitted_world_model_update": True,
                                },
                            )
                        except Exception:
                            pass
                    if self.structured_evidence_memory is not None:
                        try:
                            self.structured_evidence_memory.observe_action_result(
                                self._prev[0],
                                int(aid),
                                data,
                                g1,
                                level_before=int(self._prev_level),
                                level_after=int(_level_of(latest)),
                                provenance={
                                    "source": "E3AgentPolicy.next_move.after_transition",
                                    "admitted_world_model_update": True,
                                },
                            )
                        except Exception:
                            pass
                    self._observe_active_probe_transition(transition)
                    self._maybe_route_from_transitions()
            except Exception:
                # A degenerate/empty frame (e.g. shape (0,) -- the same class of
                # post-terminal sentinel diagnosed in the g50t apply_g50t_label
                # incident) makes grid_of(latest) 1-D, so grid.shape unpacks to
                # only one value. Skip recording this one transition rather than
                # crash the whole game -- the boundary_events handler just below
                # already applies this exact same defensive pattern for the same
                # reason; this mirrors it for consistency.
                pass
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
                        self._prev_level = _level_of(latest)
                        if self.structured_evidence_memory is not None:
                            try:
                                self.structured_evidence_memory.observe_action_candidate(
                                    int(mv[0]),
                                    mv[1],
                                    provenance={
                                        "source": "E3AgentPolicy.next_move.selected_action",
                                        "phase": self.phase,
                                    },
                                )
                            except Exception:
                                pass
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

        # REQ-ARC-WMTE-5717: reset the DEV-ONLY playbook-exemplar flag on the CACHED proposer so a
        # prior stall induction's injection never leaks into this call's program-synthesis-filter or
        # level-up-reinduction path. The stall fallthrough below re-arms it (stall-scoped) when gated.
        if self.proposer is not None:
            self.proposer.include_playbook_exemplars = False

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
            # STRUCTURED NAV INDUCER (REQ-ARC-WMTE-5842, opt-in CARNOT_ARC_STRUCTURED_NAV=1). For the
            # 4-direction NAVIGATION family, a hand-STRUCTURED InducedNavWorldModel -- fits per-action
            # displacement + avatar + goal from the agent's OWN transitions, correct-by-construction, NOT LLM
            # induction (which the 2026-07-20 induction-quality diagnosis found near-universally wrong,
            # heldout ~0.0) -- gives plan_in_model a CORRECT model. Gated by the SAME >=0.5 held-out trust bar
            # the other engine paths use; only fires when it actually fits a nav game (nonempty displacement +
            # a goal colour) AND earns trust AND plans a win. Non-fatal; falls through to the existing paths
            # otherwise. This is the "mechanic-class prior" the diagnosis flagged as highest-leverage, and it
            # makes arc_nav_world_model live-path-reachable (previously orphaned). Proven offline: with the
            # plan_in_model regression fix (REQ-ARC-WMTE-5841), this reaches a real tu93 level-up.
            if os.environ.get("CARNOT_ARC_STRUCTURED_NAV") == "1":
                try:
                    from carnot.agentic.arc_nav_world_model import InducedNavWorldModel

                    nav = InducedNavWorldModel.fit(active_transitions)
                    # CONFIDENCE-GATED (REQ-ARC-WMTE-5844): only fire on a HIGH-confidence nav fit -- the
                    # fitter also fits a (spurious) model for source-verified NON-nav games (sk48 two-snake
                    # sequence-match; wa30 Sokoban crate-push), where firing installs a plan that cannot win
                    # and wastes actions. is_confident_nav rejects those (avatar captured padding-0, or <3
                    # directions) while keeping tu93.
                    is_nav = bool(getattr(nav, "displacement", None)) and (
                        getattr(nav, "goal_color", None) is not None
                    )
                    # Pass root_grid so the gate ALSO requires the goal colour present at plan-start
                    # (REQ-ARC-WMTE-5883): otherwise a goal absent from the start grid reads as already-won
                    # and plan_in_model returns a bogus ~1-step 'win' that wastes a real action.
                    is_confident = is_nav and nav.is_confident_nav(grid=self.root_grid)
                    attempt["structured_nav_is_nav_game"] = bool(is_nav)
                    attempt["structured_nav_confident"] = bool(is_confident)
                    is_nav = is_confident
                    if is_nav and self.root_grid is not None:
                        nav_eng, nav_isdone = nav.as_callables()
                        # Record the standard metrics for diagnostics, but do NOT gate on the full-grid
                        # exact-match heldout: a CORRECT avatar-only nav model scores heldout ~0 (it models
                        # the avatar's move, not the co-moving key / step-counter HUD / rails), yet
                        # plan_in_model + execution reach a REAL level-up (REQ-ARC-WMTE-5841 proved tu93 L1).
                        # Gating on >=0.5 heldout REJECTED the correct model before planning (the exact
                        # false-negative wall the 2026-07-20 diagnosis §3 named). Gate instead on nav-game-fit
                        # + plan_found; the real level counter on execution is the oracle (a wrong plan simply
                        # fails to level up, at bounded action cost, same as any engine path).
                        nav_vr = e3.WorldModelVerifier(active_transitions).score(nav_eng)
                        attempt["structured_nav_heldout"] = round(float(nav_vr.accuracy), 4)
                        attempt["structured_nav_cell_recall"] = round(float(nav_vr.cell_recall), 4)
                        _nav_diag: dict = {}
                        nav_plan = self._call_plan_in_model(
                            e3.plan_in_model,
                            nav_eng,
                            nav_isdone,
                            self.root_grid,
                            diagnostics=_nav_diag,
                            goal_energy_override=nav.goal_energy,  # REQ-ARC-WMTE-5845: nav-specific best-first
                        )
                        attempt["structured_nav_plan_diagnostics"] = _nav_diag
                        if nav_plan:
                            self._install_goal_bias(nav_isdone)
                            self.plan = nav_plan
                            attempt["planned"] = True
                            attempt["plan_length"] = len(nav_plan)
                            attempt["engine_source"] = "structured_nav_induced"
                            return
                except Exception as _nav_e:
                    attempt["structured_nav_error"] = repr(_nav_e)[:120]

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
                    _plan_diag: dict = {}
                    _plan = self._call_plan_in_model(
                        e3.plan_in_model,
                        _eng,
                        _isdone,
                        self.root_grid,
                        diagnostics=_plan_diag,
                    )
                    attempt["ttt_prior_engine_plan_diagnostics"] = _plan_diag
                    # REQ-ARC-FCP-5699-38 fix, found via a real post-submission regression
                    # investigation: this call used to install _isdone as a goal bias
                    # UNCONDITIONALLY, before planning even ran. A real repro (sc25) found this
                    # installing a bias whose energy NEVER improved across a genuine 20009-node
                    # search (initial_goal_energy=1.0, min_goal_energy_observed=1.0,
                    # termination_reason="max_nodes_reached") -- a degenerate goal predicate.
                    # Moved AFTER plan_in_model and gated on ITS OWN real search outcome (planned,
                    # or the goal energy improved at least once) instead of a SEPARATE bounded-BFS
                    # satisfiability probe -- cheaper (no extra search) and lower false-negative
                    # risk than the plain-path fix above (which the test suite caught rejecting a
                    # genuinely-reachable goal on an imperfect engine), since it reuses evidence
                    # plan_in_model ALREADY computed rather than a second, differently-bounded
                    # search that could disagree with it. Missing/absent diagnostic fields (a
                    # planner that doesn't support goal-energy search) default to installing,
                    # preserving old behavior when this signal isn't available -- only POSITIVE
                    # evidence of a flat/non-improving search suppresses the install.
                    install_bias = True
                    if _plan_diag.get("used_goal_energy_search"):
                        initial_energy = _plan_diag.get("initial_goal_energy")
                        min_energy = _plan_diag.get("min_goal_energy_observed")
                        if (
                            initial_energy is not None
                            and min_energy is not None
                            and float(min_energy) >= float(initial_energy)
                        ):
                            install_bias = False
                    attempt["ttt_prior_goal_bias_installed"] = bool(_plan or install_bias)
                    if _plan or install_bias:
                        self._install_goal_bias(_isdone)
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
                    min_goal_predicate_consistency=1.0,
                    previous_level_complete_grid=self._previous_level_complete_grid,
                    enable_subgoal_search=self.subgoal_search,
                    subgoal_budget=self.subgoal_budget,
                    value_head=self.value_head,
                    enable_factored_planner=self.factored_planner,
                    factored_trust_threshold=self.factored_trust_threshold,
                    structural_goal_provider=structural_goal_provider,
                    # LEVER #2 (REQ-ARC-WMTE-5593-4, default off): grade the induced win-predicate
                    # against the prior-level win-state exemplar so the goal veto is not inert at a
                    # deepening boundary. Live scored default unchanged unless the env flag is set.
                    goal_exemplar_grading=(
                        os.environ.get("CARNOT_ARC_GOAL_EXEMPLAR_GRADING") == "1"
                    ),
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
                # REQ-ARC-FCP-5699-38 fix: only install a goal bias the refinement loop's OWN
                # diagnostic considers satisfiable. Found via a real post-submission regression
                # investigation: an induced-but-UNSATISFIABLE goal_predicate (goal_predicate_
                # satisfiable=False, i.e. the predicate is never true for any observed state) was
                # still being installed unconditionally, biasing ALL subsequent exploration toward
                # a goal the agent's own diagnostics say is unachievable -- actively worse than no
                # bias at all. Applies here too (not just the stall-path call site below) since the
                # same nonsensical-if-unsatisfiable risk exists regardless of which branch reaches
                # this code; there is no principled reason an unsatisfiable predicate is safe to
                # install after a level-up but not on first contact.
                if outcome.goal_predicate is not None and outcome.goal_predicate_satisfiable:
                    self._install_goal_bias(outcome.goal_predicate)
                if outcome.planned:
                    self.plan = list(outcome.plan)
                return
            self._fit_dsl_model()
            # REQ-ARC-WMTE-5717/5718: STALL / first-contact path only (the level_up_reinduction
            # branch returned above). Arm the DEV-ONLY playbook injection on the cached proposer.
            # RETRIEVAL (5718) takes precedence when gated on: inject the top-K patterns relevant to
            # THIS stuck situation; fall back to the STATIC block (5717) if retrieval is unavailable,
            # else nothing. Default (both unset) -> False -> byte-identical prompt.
            injection: bool | str = False
            injection_mode = "none"
            if _playbook_retrieval_gate_on():
                block = self._retrieve_playbook_block(active_transitions)
                if block:
                    injection, injection_mode = block, "retrieval"
                elif _playbook_exemplars_gate_on():
                    injection, injection_mode = True, "static_retrieval_unavailable"
            elif _playbook_exemplars_gate_on():
                injection, injection_mode = True, "static"
            self._proposer().include_playbook_exemplars = injection
            attempt["playbook_exemplars_injected"] = bool(injection)
            attempt["playbook_injection_mode"] = injection_mode
            # Graduated to default-on (REQ-ARC-FCP-5699-35, was DEV-ONLY per REQ-ARC-FCP-5699-24
            # / -25): the refactor/refinement loop (execute_bounded_llm_reinduction) was
            # previously reachable ONLY from the level_up_reinduction branch above, so
            # stall-triggered first-contact induction got exactly one shot with zero
            # counterexample-driven refinement. This routes the SAME bounded-refinement
            # mechanism through the stall path, with previous_level_complete_grid=None and
            # structural_goal_provider=None (both confirmed handled gracefully by
            # execute_bounded_llm_reinduction without crashing -- the goal-repair half still
            # can't help without an exemplar, per REQ-ARC-FCP-5699-24, but the DYNAMICS-side
            # refactor rounds operate on transition mismatches, which DO exist pre-first-win).
            # CARNOT_ARC_STALL_REFACTOR_LOOP=0 remains an explicit opt-out, matching the
            # CARNOT_ARC_MTP=0 pattern in _proposer() above.
            #
            # IMPORTANT graduation fix (REQ-ARC-FCP-5699-35): the dev-only version below
            # unconditionally `return`ed after this block regardless of whether stall_outcome
            # actually reached a plan. That was fine while opt-in (an operator explicitly chose
            # to trade the active_probe/plain-path fallback away), but silently discarding a
            # working fallback for every non-planned outcome would be a real regression once this
            # is the production default -- REQ-ARC-FCP-5699-32's own measurement showed the
            # min_heldout_accuracy=1.0 gate is rarely met (0/6 rounds across a full real run on
            # g50t). So this now falls through to the unchanged active_probe_controller / plain
            # single-shot path below whenever the refinement loop does NOT reach a planned
            # outcome, instead of returning unconditionally -- strictly additive, never removes
            # the pre-graduation fallback capability.
            if os.environ.get("CARNOT_ARC_STALL_REFACTOR_LOOP", "1") != "0":
                # Local try/except (REQ-ARC-FCP-5699-35 hardening, found via a real regression
                # while graduating: test_req_arc_wmte_4494_live_policy_uses_trust_energy_candidate
                # broke because execute_bounded_llm_reinduction's round-2 refactor() call can
                # raise (e.g. a proposer/model that can't complete a refactor round) BEFORE ever
                # returning a result -- an exception here previously propagated straight to this
                # method's OUTERMOST except block, aborting the ENTIRE induce-and-plan attempt
                # and silently skipping the plain single-shot / trust-energy-selector path below,
                # not just the planned=True short-circuit my fallthrough already handles. Any
                # exception during the bounded-refinement attempt now falls through exactly like
                # a clean planned=False outcome -- strictly additive, never removes the
                # pre-graduation fallback capability, matching the surrounding
                # program_synthesis_filter_error / ttt_prior_engine_error non-fatal pattern.
                try:
                    stall_outcome = execute_bounded_llm_reinduction(
                        game=self.short,
                        transitions=active_transitions,
                        cell=self.cell,
                        root_grid=self.root_grid,
                        proposer=self._proposer(),
                        candidate_provider=self._world_model_candidates,
                        load_engine=e3.load_engine,
                        plan_in_model=self._guided_plan_in_model(e3.plan_in_model),
                        max_rounds=MAX_REFINEMENT_ROUNDS,
                        min_heldout_accuracy=1.0,
                        min_goal_predicate_consistency=1.0,
                        previous_level_complete_grid=self._previous_level_complete_grid,
                        enable_subgoal_search=self.subgoal_search,
                        subgoal_budget=self.subgoal_budget,
                        value_head=self.value_head,
                        enable_factored_planner=self.factored_planner,
                        factored_trust_threshold=self.factored_trust_threshold,
                        structural_goal_provider=None,
                        goal_exemplar_grading=(
                            os.environ.get("CARNOT_ARC_GOAL_EXEMPLAR_GRADING") == "1"
                        ),
                    )
                except Exception as stall_exc:
                    attempt["stall_refactor_loop_used"] = False
                    attempt["stall_refactor_loop_error"] = repr(stall_exc)[:160]
                    stall_outcome = None
                if stall_outcome is not None:
                    attempt.update(
                        {
                            "model_specs": stall_outcome.model_specs,
                            "planned": bool(stall_outcome.planned),
                            "skipped": stall_outcome.skipped,
                            "plan_length": len(stall_outcome.plan),
                            "selected_candidate_name": stall_outcome.selected_candidate_name,
                            "refinement_rounds_used": int(stall_outcome.refinement_rounds_used),
                            "refinement_rounds": list(stall_outcome.rounds),
                            "counterexamples": list(stall_outcome.counterexamples),
                            "verifier_is_oracle": bool(stall_outcome.verifier_is_oracle),
                            "goal_predicate_satisfiable": bool(
                                stall_outcome.goal_predicate_satisfiable
                            ),
                            "goal_satisfiability": dict(stall_outcome.goal_satisfiability),
                            "stall_refactor_loop_used": True,
                        }
                    )
                    # REQ-ARC-FCP-5699-38 fix (see the twin fix + full rationale at the
                    # level_up_reinduction call site above): only install a goal bias the
                    # refinement loop's own diagnostic considers satisfiable. This is the call
                    # site a REAL post-submission investigation found live-firing on genuine
                    # first-contact stalls: goal_predicate_satisfiable=False, planned=False,
                    # skipped="hidden_state_trust_below_threshold", yet a goal bias was still
                    # being installed and persisting into the rest of the episode's exploration.
                    if (
                        stall_outcome.goal_predicate is not None
                        and stall_outcome.goal_predicate_satisfiable
                    ):
                        self._install_goal_bias(stall_outcome.goal_predicate)
                    if stall_outcome.planned:
                        self.plan = list(stall_outcome.plan)
                        return
                # else: fall through to active_probe_controller / plain single-shot path below
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
            # REQ-ARC-FCP-5699-38 (plain single-shot path): found via the same post-submission
            # regression investigation that fixed the two stall-refactor-loop call sites above --
            # this pre-existing (not introduced by REQ-35) code path installs a goal bias whenever
            # DYNAMICS trust passes (world-model prediction accuracy above threshold), with NO
            # check that the GOAL predicate itself is satisfiable -- a well-predicted world model
            # can still be paired with a goal that is never true for any reachable state.
            #
            # DEV-ONLY, opt-in (unset -> byte-identical to before this REQ): a real test run
            # (test_req_arc_wmte_4494_live_policy_uses_trust_energy_candidate) found this bounded-
            # BFS check, reused unmodified from execute_bounded_llm_reinduction, produces a real
            # false negative against a deliberately-simplified engine -- correctly-per-its-own-
            # logic determining a goal unreachable within the search bound, when the goal WAS
            # trivially reachable in that test's simplified world. This means an IMPERFECT (but
            # still useful) induced engine could analogously fail to prove a genuinely-achievable
            # goal reachable within max_nodes/max_depth, rejecting a good goal, not just a bad one
            # -- a real design tension the two already-graduated fixes above do not share (those
            # gate on a value execute_bounded_llm_reinduction ALREADY computes and had already
            # gone through the same REQ-25->32->35 dev-gated validation cycle; this is brand-new
            # computation on a path that never paid this cost before). Needs a real matched-budget
            # validation pass before graduating to default-on, per this project's own standing
            # discipline for exactly this class of change -- not shipped as an unconditional
            # default on the strength of one investigation session.
            if os.environ.get("CARNOT_ARC_PLAIN_PATH_GOAL_SATISFIABILITY_CHECK") == "1":
                goal_check = _goal_satisfiability_check(
                    engine=engine, goal=is_done, start_grid=self.root_grid
                )
                attempt["goal_predicate_satisfiable"] = bool(goal_check.get("satisfiable"))
                attempt["goal_satisfiability"] = dict(goal_check)
                if not goal_check.get("satisfiable"):
                    attempt["skipped"] = "degenerate_goal_predicate"
                    return
            self._install_goal_bias(is_done)
            # plan ENTIRELY in the model (zero real actions); execute phase RESETs then
            # replays this plan in the real env, halting on divergence.
            _plan_diag2: dict = {}
            plan = self._call_plan_in_model(
                e3.plan_in_model, engine, is_done, self.root_grid, diagnostics=_plan_diag2
            )
            attempt["plan_diagnostics"] = _plan_diag2
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
    # REQ-ARC-WMTE-5836: the two just-explore frontier-discipline mechanisms + the
    # within-tier draw knob.
    #
    # STATUS CORRECTION (2026-07-25): this comment used to read "All THREE default OFF ->
    # the submitted agent is unchanged until the matched-budget offline A/B greenlights a
    # flip". That is no longer true -- the A/B ran (results/experiment_5836_frontier_
    # discipline_generalization.json) and tier_exhaustion + tier_uniform_random +
    # tier_click_vocab_only were flipped ON, so the submitted agent HAS changed. Only
    # frontier_distance_gradient is still off. Left as a note rather than deleted because a
    # stale "nothing has changed" claim in the config's own documentation is exactly the
    # kind of drift that makes a later reader mis-attribute a measurement.
    "frontier_tier_exhaustion": SUBMITTED_FRONTIER_TIER_EXHAUSTION_ENABLED,
    "frontier_tier_exhaustion_mode": SUBMITTED_FRONTIER_TIER_EXHAUSTION_MODE,
    "frontier_tier_count": SUBMITTED_FRONTIER_TIER_COUNT,
    "frontier_tier_uniform_random": SUBMITTED_FRONTIER_TIER_UNIFORM_RANDOM_ENABLED,
    "frontier_tier_click_vocab_only": SUBMITTED_FRONTIER_TIER_CLICK_VOCAB_ONLY_ENABLED,
    "frontier_distance_gradient": SUBMITTED_FRONTIER_DISTANCE_GRADIENT_ENABLED,
    "frontier_distance_gradient_mode": SUBMITTED_FRONTIER_DISTANCE_GRADIENT_MODE,
    # REQ-ARC-WMTE-5950: per-object click-pixel sampling. DEFAULT OFF -> the submitted
    # agent's click coordinates are byte-identical to today's until arm F beats arm B2
    # (the CURRENT live configuration) in a matched-budget offline A/B.
    "click_pixel_sampling": SUBMITTED_CLICK_PIXEL_SAMPLING_ENABLED,
    "click_pixel_samples_per_component": SUBMITTED_CLICK_PIXEL_SAMPLES_PER_COMPONENT,
    "click_pixel_redraw_budget": SUBMITTED_CLICK_PIXEL_REDRAW_BUDGET,
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
    "object_history_salience_enabled": SUBMITTED_OBJECT_HISTORY_SALIENCE_ENABLED,
    "program_synthesis_proposal_filter_enabled": (
        SUBMITTED_PROGRAM_SYNTHESIS_PROPOSAL_FILTER_ENABLED
    ),
    "program_synthesis_proposal_filter_trust_threshold": (
        SUBMITTED_PROGRAM_SYNTHESIS_PROPOSAL_FILTER_TRUST_THRESHOLD
    ),
    "inert_click_pruner_enabled": SUBMITTED_INERT_CLICK_PRUNER_ENABLED,
    "amortized_first_contact_prior_enabled": SUBMITTED_AMORTIZED_FIRST_CONTACT_PRIOR_ENABLED,
    "amortized_first_contact_prior_mode": SUBMITTED_AMORTIZED_FIRST_CONTACT_PRIOR_MODE,
    "go_explore_archive_enabled": SUBMITTED_GO_EXPLORE_ARCHIVE_ENABLED,
    "go_explore_archive_mode": SUBMITTED_GO_EXPLORE_ARCHIVE_MODE,
    "ige_cell_selection_enabled": SUBMITTED_IGE_CELL_SELECTION_ENABLED,
    "ige_cell_selection_mode": SUBMITTED_IGE_CELL_SELECTION_MODE,
    "matm_similarity_retrieval_enabled": SUBMITTED_MATM_SIMILARITY_RETRIEVAL_ENABLED,
    "matm_similarity_retrieval_mode": SUBMITTED_MATM_SIMILARITY_RETRIEVAL_MODE,
    "epistemic_ledger_enabled": SUBMITTED_EPISTEMIC_LEDGER_ENABLED,
    "epistemic_ledger_mode": SUBMITTED_EPISTEMIC_LEDGER_MODE,
    "structured_evidence_memory_enabled": SUBMITTED_STRUCTURED_EVIDENCE_MEMORY_ENABLED,
    "structured_evidence_memory_mode": SUBMITTED_STRUCTURED_EVIDENCE_MEMORY_MODE,
    "auto_hud_mask_enabled": SUBMITTED_AUTO_HUD_MASK_ENABLED,
    "auto_hud_mask_mode": SUBMITTED_AUTO_HUD_MASK_MODE,
    "edge_bar_hud_mask_enabled": SUBMITTED_EDGE_BAR_HUD_MASK_ENABLED,
    "edge_bar_hud_mask_mode": SUBMITTED_EDGE_BAR_HUD_MASK_MODE,
    "hud_mask_collapse_guard_enabled": SUBMITTED_HUD_MASK_COLLAPSE_GUARD_ENABLED,
    "hud_mask_collapse_guard_mode": SUBMITTED_HUD_MASK_COLLAPSE_GUARD_MODE,
    "router_wired": True,
    "solve_learning_router_wired": True,
    "strategy_router_enabled": True,
    "discriminative_router_wired": True,
    "discriminative_candidate_router_enabled": True,
    "candidate_router": "cross_game_discriminative_v3_tiebreaker",
    "sge_candidate_router_wired": True,
    "sge_candidate_router_enabled": SUBMITTED_SGE_CANDIDATE_ROUTER_ENABLED,
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
