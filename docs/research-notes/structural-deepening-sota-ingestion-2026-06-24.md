# Structural-deepening SOTA ingestion 2026-06-24

```json
{
  "citations_verified": {
    "1011.0686": {
      "http_status": 200,
      "title": "A Reduction of Imitation Learning and Structured Prediction to No-Regret Online Learning",
      "url": "https://arxiv.org/abs/1011.0686"
    },
    "2504.04366": {
      "http_status": 200,
      "title": "Solving Sokoban using Hierarchical Reinforcement Learning with Landmarks",
      "url": "https://arxiv.org/abs/2504.04366"
    },
    "2505.10819": {
      "http_status": 200,
      "title": "PoE-World: Compositional World Modeling with Products of Programmatic Experts",
      "url": "https://arxiv.org/abs/2505.10819"
    },
    "2506.07255": {
      "http_status": 200,
      "title": "Subgoal-Guided Policy Heuristic Search with Learned Subgoals",
      "url": "https://arxiv.org/abs/2506.07255"
    },
    "2604.03208": {
      "http_status": 200,
      "title": "Hierarchical Planning with Latent World Models",
      "url": "https://arxiv.org/abs/2604.03208"
    },
    "2604.11351": {
      "http_status": 200,
      "title": "WM-DAgger: Enabling Efficient Data Aggregation for Imitation Learning with World Models",
      "url": "https://arxiv.org/abs/2604.11351"
    },
    "2605.05138": {
      "http_status": 200,
      "title": "Executable World Models for ARC-AGI-3 in the Era of Coding Agents",
      "url": "https://arxiv.org/abs/2605.05138"
    },
    "2605.12913": {
      "http_status": 200,
      "title": "Revisiting DAgger in the Era of LLM-Agents",
      "url": "https://arxiv.org/abs/2605.12913"
    }
  },
  "deep_research_not_used": true,
  "field_principles": {
    "citations_verified": {
      "principle": "each cited arXiv ID with an HTTP-200 verification -- pre-empts fabricated citations."
    },
    "deep_research_not_used": {
      "principle": "MUST be true -- /deep-research is BANNED in the autonomous loop; used sweep helpers + low-concurrency WebSearch/WebFetch."
    },
    "flagged_for_next_roadmap": {
      "principle": "the strongest method(s) flagged as candidate .431 inputs (flagged_for_v431) -- closes discover->ingest->plan->experiment."
    },
    "honest_verdict": {
      "principle": "terminal prefix; success: sota_ingestion_structural_deepening_mapped."
    },
    "inference_substrate": {
      "principle": "aggregation_from_upstream_artifacts -- literature read + synthesis, no model load (100us floor)."
    },
    "methods_mapped": {
      "principle": "the strongest 3-5 SOTA methods with REAL arXiv IDs + per-method implement-cost-over-current-stack + fails_when (no citation = fabrication)."
    },
    "note_path": {
      "principle": "the per-track research-note path (the SOTA-Ingestion Cycle deliverable)."
    },
    "preconditions_checked": {
      "principle": "records network reachability verified; pre-empts fabricated citations."
    }
  },
  "flagged_for_next_roadmap": [
    "flagged_for_v431: hierarchical_subgoal_e3_frontier_with_a1_a2_tiebreakers (arXiv:2604.03208 + arXiv:2506.07255 + arXiv:2504.04366 + arXiv:2605.12913 + arXiv:1011.0686)",
    "flagged_for_v431: poe_world_factored_executable_subgoal_planner (arXiv:2505.10819 + arXiv:2605.05138)"
  ],
  "honest_verdict": "success: sota_ingestion_structural_deepening_mapped",
  "inference_substrate": "aggregation_from_upstream_artifacts",
  "methods_mapped": [
    {
      "fails_when": "the subgoal layer proposes visual states that are not mechanically goal relevant, A2 still cannot separate live frontier states, bounded search cannot reach the proposed subgoal, or live E3 replay rejects the path.",
      "implement_cost_over_current_stack": "medium-high: add a high-level subgoal layer above the current live E3 frontier, mine A1/A2 failed search traces for subgoal candidates, run bounded low-level search per subgoal, and keep replay plus matched no-regression controls.",
      "maps_to_current_stack": "A1 L2-goal-induction becomes a subgoal proposer instead of one global terminal predicate, A2 distribution-corrected value-routing becomes the tie-breaker inside each bounded subgoal search, and live E3 remains the replay-verified executor.",
      "method": "Hierarchical subgoal search over the live E3 frontier",
      "residual_scope": "A1 residual single_exemplar_goal_insufficient left unsatisfiable L2 goal predicates and empty plans; A2 residual missing_verifier_gap_live_frontier_not_separated left zero live lift after distribution-corrected value routing.",
      "roadmap_candidate": "flagged_for_v431: hierarchical_subgoal_e3_frontier_with_a1_a2_tiebreakers (arXiv:2604.03208 + arXiv:2506.07255 + arXiv:2504.04366 + arXiv:2605.12913 + arXiv:1011.0686)",
      "source_ids": [
        "2604.03208",
        "2506.07255",
        "2504.04366",
        "2605.05138"
      ],
      "track": "hierarchical_subgoal_search_live_e3_frontier"
    },
    {
      "fails_when": "failed search trees contain no reusable near-goal states, labels are too sparse to choose among subgoals, or the value head only reshuffles a candidate set without any mechanically valid L2 action.",
      "implement_cost_over_current_stack": "medium: retain failed A1/A2 frontier trees, label promising partial states from replay and value deltas, train a subgoal-conditioned proposal table, and use the corrected value head only at decision points where candidates are otherwise tied.",
      "maps_to_current_stack": "A1 L2-goal-induction supplies candidate post-level-up goal states, A2 distribution-corrected value-routing ranks tree-local alternatives rather than every primitive action globally, and live E3 supplies the failed trees plus replay adjudication.",
      "method": "Failed-search-tree subgoal proposer with value tie-breaking",
      "residual_scope": "A1 residual single_exemplar_goal_insufficient left unsatisfiable L2 goal predicates and empty plans; A2 residual missing_verifier_gap_live_frontier_not_separated left zero live lift after distribution-corrected value routing.",
      "roadmap_candidate": "flagged_for_v431: hierarchical_subgoal_e3_frontier_with_a1_a2_tiebreakers (arXiv:2604.03208 + arXiv:2506.07255 + arXiv:2504.04366 + arXiv:2605.12913 + arXiv:1011.0686)",
      "source_ids": [
        "2506.07255",
        "2605.12913",
        "1011.0686"
      ],
      "track": "failed_search_tree_subgoal_proposer"
    },
    {
      "fails_when": "expert factors are not independent, rare interactions are smoothed away by the product, generated experts overfit prefix transitions, or product-model plans emit actions that live E3 cannot replay.",
      "implement_cost_over_current_stack": "medium-high: induce small programmatic experts for object-level preconditions and effects, weight experts by held-out transition trust, compose only replay-stable factors, and plan subgoal-conditioned sequences through the product model.",
      "maps_to_current_stack": "A1 L2-goal-induction proposes the subgoal predicates each expert must make reachable, A2 distribution-corrected value-routing scores which expert-predicted states deserve live expansion, and live E3 executes and audits every emitted plan.",
      "method": "PoE-World factored executable model subgoal planner",
      "residual_scope": "A1 residual single_exemplar_goal_insufficient left unsatisfiable L2 goal predicates and empty plans; A2 residual missing_verifier_gap_live_frontier_not_separated left zero live lift after distribution-corrected value routing.",
      "roadmap_candidate": "flagged_for_v431: poe_world_factored_executable_subgoal_planner (arXiv:2505.10819 + arXiv:2605.05138)",
      "source_ids": [
        "2505.10819",
        "2605.05138"
      ],
      "track": "poe_world_factored_executable_model_planner"
    },
    {
      "fails_when": "the executable world model hallucinates OOD recovery transitions, trust weights accept brittle experts, subgoal partitions are too small for value calibration, or the calibrated value still sees no valid L2 path.",
      "implement_cost_over_current_stack": "medium: aggregate live-frontier states under each proposed subgoal, synthesize or replay OOD recovery transitions only when the executable world model is held-out trusted, and calibrate value scores as bounded subgoal-local costs.",
      "maps_to_current_stack": "A1 L2-goal-induction defines the subgoal-conditioned state distribution, A2 distribution-corrected value-routing is retrained or calibrated on that distribution, and live E3 provides both the frontier states and the final replay gate.",
      "method": "WM-DAgger trust-weighted subgoal-conditioned value routing",
      "residual_scope": "A1 residual single_exemplar_goal_insufficient left unsatisfiable L2 goal predicates and empty plans; A2 residual missing_verifier_gap_live_frontier_not_separated left zero live lift after distribution-corrected value routing.",
      "roadmap_candidate": "flagged_for_v431: hierarchical_subgoal_e3_frontier_with_a1_a2_tiebreakers (arXiv:2604.03208 + arXiv:2506.07255 + arXiv:2504.04366 + arXiv:2605.12913 + arXiv:1011.0686)",
      "source_ids": [
        "2605.12913",
        "2604.11351",
        "1011.0686",
        "2605.05138"
      ],
      "track": "wm_dagger_trust_weighted_subgoal_value_routing"
    }
  ],
  "note_path": "docs/research-notes/structural-deepening-sota-ingestion-2026-06-24.md",
  "preconditions_checked": {
    "agents_md_read": true,
    "arxiv_http_200_verified_ids": [
      "https://arxiv.org/abs/1011.0686",
      "https://arxiv.org/abs/2504.04366",
      "https://arxiv.org/abs/2505.10819",
      "https://arxiv.org/abs/2506.07255",
      "https://arxiv.org/abs/2604.03208",
      "https://arxiv.org/abs/2604.11351",
      "https://arxiv.org/abs/2605.05138",
      "https://arxiv.org/abs/2605.12913"
    ],
    "codex_md_read": true,
    "deep_research_invoked": false,
    "exp4661_artifact_read": true,
    "exp4661_note_read": true,
    "exp4664_artifact_read": true,
    "exp4665_artifact_read": true,
    "leaderboard_submission": false,
    "live_llm_inference": false,
    "model_load": false,
    "network_hf_models_reachable": true,
    "ops_docs_modified": false,
    "research_conductor_modified": false,
    "research_references_read": true,
    "research_studying_read": true,
    "sweep_clusters_help_ok": true,
    "sweep_clusters_urls": [
      "http://export.arxiv.org/api/query?search_query=(abs:\"neural+guided+search\"+OR+abs:\"learned+heuristic\"+OR+abs:\"value+guided+search\"+OR+abs:\"program+induction\"+OR+abs:\"world+model\"+OR+abs:\"goal+induction\")+AND+(abs:\"planning\"+OR+abs:\"agent\"+OR+abs:\"reasoning\"+OR+abs:\"reinforcement+learning\")&start=0&max_results=8&sortBy=submittedDate&sortOrder=descending",
      "http://export.arxiv.org/api/query?search_query=(abs:\"affordance\"+OR+abs:\"action+effect\"+OR+abs:\"clickability\"+OR+abs:\"frame+prediction\"+OR+abs:\"intrinsic+motivation\"+OR+abs:\"directed+exploration\"+OR+abs:\"novelty+search\")+AND+(abs:\"reinforcement+learning\"+OR+abs:\"agent\"+OR+abs:\"exploration\"+OR+abs:\"interactive+environment\"+OR+abs:\"ARC\")&start=0&max_results=8&sortBy=submittedDate&sortOrder=descending"
    ],
    "sweep_clusters_used": true,
    "sweep_semscholar_arxiv_ids": [],
    "sweep_semscholar_queries": [
      "hierarchical subgoal search learned subgoals latent world model ARC 2604.03208 2506.07255 2504.04366",
      "factored executable world model product of experts programmatic experts ARC 2505.10819 2605.05138",
      "subgoal conditioned planning DAgger distribution shift value routing live frontier 2605.12913 1011.0686"
    ],
    "sweep_semscholar_rate_limited_queries": [
      "hierarchical subgoal search learned subgoals latent world model ARC 2604.03208 2506.07255 2504.04366",
      "factored executable world model product of experts programmatic experts ARC 2505.10819 2605.05138",
      "subgoal conditioned planning DAgger distribution shift value routing live frontier 2605.12913 1011.0686"
    ],
    "sweep_semscholar_used": true,
    "training_launched": false,
    "websearch_webfetch_top_sources": [
      "https://arxiv.org/abs/2604.03208",
      "https://arxiv.org/abs/2506.07255",
      "https://arxiv.org/abs/2504.04366",
      "https://arxiv.org/abs/2505.10819",
      "https://arxiv.org/abs/2605.05138",
      "https://arxiv.org/abs/2605.12913",
      "https://arxiv.org/abs/2604.11351",
      "https://arxiv.org/abs/1011.0686"
    ]
  },
  "random_seed": 4673
}
```

## Fresh-pass provenance

Read `AGENTS.md`, `CODEX.md`,
`results/experiment_4661_sota_ingestion_generation_guidance.json`,
`docs/research-notes/generation-guidance-sota-ingestion-2026-06-24.md`,
`results/experiment_4664_l2_goal_predicate_induction_live.json`,
`results/experiment_4665_dagger_distribution_shift_value_routing.json`,
`research-studying.md`, and `research-references.md`. The current stack is A1
L2-goal-induction plus A2 distribution-corrected value-routing inside the live
E3 path. A1 closed with `single_exemplar_goal_insufficient`: the induced L2
goals were unsatisfiable, plans were length zero, and no L2 plan reached the
goal. A2 closed with `missing_verifier_gap_live_frontier_not_separated`: the
distribution shift score was corrected from 0.699108 to 0.0, but first-win and
solve-rate deltas were still 0.0. This pass therefore maps a structural fallback
instead of another scalar reranker.

Reliable-channel pass, not `/deep-research`:
- `curl -sf -o /dev/null https://huggingface.co/api/models`
- `.venv/bin/python scripts/sweep_clusters.py --help`
- `.venv/bin/python scripts/sweep_clusters.py 6 --max-results 8`
- `.venv/bin/python scripts/sweep_clusters.py 5 --max-results 8`
- `.venv/bin/python scripts/sweep_semscholar.py - --limit 8` with three focused queries
- low-concurrency WebSearch/WebFetch of the top hierarchical, factored-model, and DAgger papers
- direct arXiv HTTP checks for all cited IDs

Semantic Scholar returned HTTP 429 for the focused queries and no S2-only source
was promoted. Direct arXiv HTTP checks returned 200 for arXiv:2604.03208,
arXiv:2506.07255, arXiv:2504.04366, arXiv:2505.10819, arXiv:2605.05138,
arXiv:2605.12913, arXiv:2604.11351, and arXiv:1011.0686. No live LLM inference,
No training, No leaderboard submission, no model load, and no live solve claim
were run or made. `scripts/research_conductor.py`,
`ops/changelog.md`, and `ops/status.md` were not edited by this workflow.

## SOTA -> .431 structural mapping

## Hierarchical subgoal search over the live E3 frontier

**Sources:** Hierarchical Planning with Latent World Models, arXiv:2604.03208;
Subgoal-Guided Policy Heuristic Search with Learned Subgoals, arXiv:2506.07255;
Sokoban hierarchical landmarks, arXiv:2504.04366; Executable World Models for
ARC-AGI-3, arXiv:2605.05138.

**Mapping to current stack:** A1 L2-goal-induction should stop being one global
terminal predicate and become a subgoal proposer. A2 distribution-corrected
value-routing should be used as a local tie-breaker inside each bounded subgoal
search. The live E3 path remains the executor and replay gate.

**Implementation cost over current stack:** medium-high. Add subgoal mining
from failed A1/A2 trees, run bounded low-level search for each selected subgoal,
and retain matched baseline, no-subgoal ablation, random-subgoal ablation, and
replay gates.

**Fails when:** proposed subgoals are visually plausible but mechanically
irrelevant, the corrected value head still does not separate live frontier
states, bounded search cannot reach the chosen subgoal, or live E3 rejects the
path.

## Failed-search-tree subgoal proposer with value tie-breaking

**Sources:** Subgoal-guided heuristic search, arXiv:2506.07255; Revisiting
DAgger in the Era of LLM-Agents, arXiv:2605.12913; original DAgger,
arXiv:1011.0686.

**Mapping to current stack:** A1 L2-goal-induction provides candidate
post-level-up states even when the terminal goal is not yet satisfiable. A2
distribution-corrected value-routing ranks alternatives within a subgoal-local
tree, not every primitive action globally. The live E3 failed frontier is the
training and replay substrate.

**Implementation cost over current stack:** medium. Persist failed search
trees, label promising partial states from replay/value deltas, learn a
subgoal-conditioned proposal table, and use value scores only at bounded
decision points.

**Fails when:** failed trees contain no reusable near-goal states, labels are
too sparse to choose among subgoals, or tie-breaking only reshuffles candidates
that do not contain a valid L2 action.

## PoE-World factored executable model subgoal planner

**Sources:** PoE-World, arXiv:2505.10819; Executable World Models for ARC-AGI-3,
arXiv:2605.05138.

**Mapping to current stack:** A1 L2-goal-induction proposes the predicates each
factor should make reachable. A2 distribution-corrected value-routing scores
which product-model states deserve live expansion. The live E3 path executes
and audits every emitted plan. This is the strongest answer to the A2 residual
because QD needs a factored feasibility model before it mutates.

**Implementation cost over current stack:** medium-high. Induce object-level
precondition/effect experts, weight them by held-out transition trust, compose
only replay-stable factors, and search through the product model with hard live
replay checks.

**Fails when:** expert factors are not independent, rare interactions are
smoothed away, generated experts overfit prefix transitions, or product-model
planning emits a live-invalid action sequence.

## WM-DAgger trust-weighted subgoal-conditioned value routing

**Sources:** Revisiting DAgger in the Era of LLM-Agents, arXiv:2605.12913;
WM-DAgger, arXiv:2604.11351; original DAgger, arXiv:1011.0686; Executable World
Models for ARC-AGI-3, arXiv:2605.05138.

**Mapping to current stack:** A1 L2-goal-induction defines subgoal-conditioned
state distributions. A2 distribution-corrected value-routing is retrained or
calibrated on those distributions instead of one mixed frontier. The live E3
path provides the frontier states and the final replay gate.

**Implementation cost over current stack:** medium. Aggregate live-frontier
states per subgoal, synthesize or replay OOD recovery transitions only when a
trusted executable model supports them, and calibrate values as subgoal-local
costs.

**Fails when:** the executable model hallucinates recovery transitions, trust
weights accept brittle experts, subgoal partitions are too small for value
calibration, or the calibrated value still sees no valid L2 path.

## Bottom line for the .431 roadmap

1. Build `flagged_for_v431: hierarchical_subgoal_e3_frontier_with_a1_a2_tiebreakers`
   first. It directly addresses `single_exemplar_goal_insufficient` by turning
   A1 into a subgoal proposer and directly addresses
   `missing_verifier_gap_live_frontier_not_separated` by restricting A2 to
   local subgoal tie-breaking. The core citations are arXiv:2604.03208,
   arXiv:2506.07255, arXiv:2504.04366, arXiv:2605.12913, and arXiv:1011.0686.
2. Keep `flagged_for_v431: poe_world_factored_executable_subgoal_planner` as the
   second structural candidate when transition-factor trust is available.
   PoE-World arXiv:2505.10819 plus executable ARC world models
   arXiv:2605.05138 is the best factored-model answer to A2's no-winner bridge.
3. Use WM-DAgger arXiv:2604.11351 as a support mechanism only after a subgoal or
   product-model scaffold exists; it is not another standalone value-weight run.

