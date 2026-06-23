# Energy-fitness generator literature ingestion 2026-06-23

```json
{
  "citations_verified": {
    "1710.11089": {
      "http_status": 200,
      "title": "Eigenoption Discovery through the Deep Successor Representation",
      "url": "https://arxiv.org/abs/1710.11089"
    },
    "1810.04586": {
      "http_status": 200,
      "title": "The Laplacian in RL: Learning Representations with Efficient Approximations",
      "url": "https://arxiv.org/abs/1810.04586"
    },
    "2107.07031": {
      "http_status": 200,
      "title": "Experimental Evidence that Empowerment May Drive Exploration in Sparse-Reward Environments",
      "url": "https://arxiv.org/abs/2107.07031"
    },
    "2302.04693": {
      "http_status": 200,
      "title": "Scaling Goal-based Exploration via Pruning Proto-goals",
      "url": "https://arxiv.org/abs/2302.04693"
    },
    "2308.05483": {
      "http_status": 200,
      "title": "Quality Diversity under Sparse Reward and Sparse Interaction",
      "url": "https://arxiv.org/abs/2308.05483"
    },
    "2502.02962": {
      "http_status": 200,
      "title": "Intrinsic motivation as constrained entropy maximization",
      "url": "https://arxiv.org/abs/2502.02962"
    },
    "2504.01915": {
      "http_status": 200,
      "title": "Overcoming Deceptiveness in Fitness Optimization with Unsupervised Quality-Diversity",
      "url": "https://arxiv.org/abs/2504.01915"
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
    "2605.05138": {
      "http_status": 200,
      "title": "Executable World Models for ARC-AGI-3 in the Era of Coding Agents",
      "url": "https://arxiv.org/abs/2605.05138"
    },
    "2605.27130": {
      "http_status": 200,
      "title": "DEI: Diversity in Evolutionary Inference for Quality-Diversity Search",
      "url": "https://arxiv.org/abs/2605.27130"
    },
    "2605.28814": {
      "http_status": 200,
      "title": "Self-Improving Language Models with Bidirectional Evolutionary Search",
      "url": "https://arxiv.org/abs/2605.28814"
    }
  },
  "deep_research_not_used": true,
  "field_principles": {
    "deep_research_not_used": {
      "principle": "MUST be true -- /deep-research is banned in the autonomous loop; used sweep helpers + low-concurrency WebSearch/WebFetch."
    },
    "flagged_for_next_roadmap": {
      "principle": "the strongest method(s) flagged as candidate .429 inputs -- closes discover->ingest->plan->experiment."
    },
    "honest_verdict": {
      "principle": "terminal prefix; success: sota_ingestion_energy_fitness_generator_mapped."
    },
    "inference_substrate": {
      "principle": "aggregation_from_upstream_artifacts -- literature read + synthesis, no model load (100us floor)."
    },
    "methods_mapped": {
      "principle": "the strongest 3-5 SOTA methods with REAL arXiv IDs + per-method implement-cost-over-current-stack + fails_when -- the actionable ingestion (no citation = fabrication)."
    },
    "note_path": {
      "principle": "docs/research-notes/energy-fitness-generator-literature-2026-06-23.md -- the per-track note (the SOTA-Ingestion Cycle deliverable)."
    },
    "preconditions_checked": {
      "principle": "records network reachability verified; pre-empts fabricated citations."
    }
  },
  "flagged_for_next_roadmap": [
    "flagged_for_v429: energy_as_fitness_qd_bes_action_sequence_generator (arXiv:2605.28814 + arXiv:2308.05483 + arXiv:2504.01915)",
    "flagged_for_v429: macro_action_vocabulary_empowerment_options (arXiv:2107.07031 + arXiv:2502.02962 + arXiv:2302.04693 + arXiv:1710.11089)",
    "flagged_for_v429: hierarchical_subgoal_search_over_goal_energy (arXiv:2604.03208 + arXiv:2506.07255 + arXiv:2504.04366)",
    "flagged_for_v429: poe_world_factored_executable_model_planner (arXiv:2505.10819 + arXiv:2605.05138)",
    "flagged_for_v429: distributed_qd_mutation_ensemble_later_scaling (arXiv:2605.27130 + arXiv:2605.28814)"
  ],
  "honest_verdict": "success: sota_ingestion_energy_fitness_generator_mapped",
  "inference_substrate": "aggregation_from_upstream_artifacts",
  "methods_mapped": [
    {
      "fails_when": "the goal-energy well is wrong or too sparse, the action-effect model aliases hidden registers, archive descriptors collapse to visual novelty, or evolution spends the fixed action budget without adding a winner to the pool.",
      "implement_cost_over_current_stack": "medium: wrap current goal-energy and action-effect rollouts in a MAP-Elites-style archive over multi-action sequences, add insert/delete/swap/splice mutation plus shared-state crossover, and score fitness by goal-energy delta, action-effect cell recall, and first-win action efficiency.",
      "maps_to_current_stack": "goal-energy becomes the dense fitness signal rather than only a terminal ranker; action-effect predictions provide transition feasibility and behavior descriptors for diverse sequence niches.",
      "method": "Energy-as-fitness QD/BES action-sequence evolution",
      "roadmap_candidate": "flagged_for_v429: energy_as_fitness_qd_bes_action_sequence_generator (arXiv:2605.28814 + arXiv:2308.05483 + arXiv:2504.01915)",
      "source_ids": [
        "2605.28814",
        "2308.05483",
        "2504.01915"
      ],
      "track": "energy_as_fitness_quality_diversity_generator"
    },
    {
      "fails_when": "macros memorize a single game's geometry, empowerment rewards high-control loops with no goal progress, eigenoption/proto-goal clusters are not reachable under the live budget, or a macro hides the primitive action that the verifier needs to debug.",
      "implement_cost_over_current_stack": "medium-high: segment cached exploration traces into controllable multi-step effects, cluster them by frame delta and reachable proto-goal, estimate empowerment/control, then expose the surviving macros through the candidate router as action-effect-backed options.",
      "maps_to_current_stack": "goal-energy selects which macro endpoints are worth reaching; action-effect validates that each macro has repeatable preconditions and effects before it is reused across games.",
      "method": "Macro-action vocabulary from empowerment, proto-goals, and eigenoptions",
      "roadmap_candidate": "flagged_for_v429: macro_action_vocabulary_empowerment_options (arXiv:2107.07031 + arXiv:2502.02962 + arXiv:2302.04693 + arXiv:1710.11089)",
      "source_ids": [
        "2107.07031",
        "2502.02962",
        "2302.04693",
        "1810.04586",
        "1710.11089"
      ],
      "track": "macro_action_vocabulary_horizon_collapse"
    },
    {
      "fails_when": "subgoals are visually plausible but off-goal, learned landmarks require complete solution trajectories the current search lacks, MCTS expands through an untrusted model, or hierarchy adds overhead without reducing primitive actions-to-win.",
      "implement_cost_over_current_stack": "medium: add a high-level subgoal planner above graph exploration, mine failed search trees for subgoals, run bounded low-level MCTS/best-first search to each subgoal, and retain the current goal-energy/action-efficiency ablation gates.",
      "maps_to_current_stack": "goal-energy becomes the high-level subgoal ranking objective; action-effect constrains low-level transitions so each subgoal is checked against executable dynamics rather than imagined screen states.",
      "method": "Hierarchical subgoal search over trusted goal-energy models",
      "roadmap_candidate": "flagged_for_v429: hierarchical_subgoal_search_over_goal_energy (arXiv:2604.03208 + arXiv:2506.07255 + arXiv:2504.04366)",
      "source_ids": [
        "2604.03208",
        "2506.07255",
        "2504.04366",
        "2605.05138"
      ],
      "track": "hierarchical_subgoal_mcts_search"
    },
    {
      "fails_when": "expert factors are not independent, the product masks a decisive rare interaction, LLM-synthesized experts overfit prefix transitions, or planning through a soft model produces actions that fail live verification.",
      "implement_cost_over_current_stack": "medium-high: induce small programmatic experts for object-level precondition/effect laws, weight them by held-out transition trust, compose them as a product model, then plan candidate macro sequences through the factored executable model.",
      "maps_to_current_stack": "goal-energy scores end states and expert-set simplicity; action-effect supplies the observed transition rows that weight and prune each programmatic expert before planning.",
      "method": "PoE-World factored executable model with plan-through-model search",
      "roadmap_candidate": "flagged_for_v429: poe_world_factored_executable_model_planner (arXiv:2505.10819 + arXiv:2605.05138)",
      "source_ids": [
        "2505.10819",
        "2605.05138"
      ],
      "track": "factored_executable_world_model_planner"
    },
    {
      "fails_when": "the base archive has no signal, model diversity only increases duplicate invalid actions, merged elites are not replay-verified, or distributed search consumes more calls without beating the single-node budget-normalized generator.",
      "implement_cost_over_current_stack": "high: after the single-node QD generator is useful, run multiple heterogeneous mutation policies over the same action-sequence archive and periodically merge elites under the shared goal-energy and action-effect verifier gates.",
      "maps_to_current_stack": "goal-energy is the common fitness contract across mutation sources; action-effect is the shared transition verifier that prevents a larger archive from admitting hallucinated action sequences.",
      "method": "Distributed QD mutation ensemble as a later scaling lever",
      "roadmap_candidate": "flagged_for_v429: distributed_qd_mutation_ensemble_later_scaling (arXiv:2605.27130 + arXiv:2605.28814)",
      "source_ids": [
        "2605.27130",
        "2605.28814"
      ],
      "track": "fresh_2026_non_ar_generation_scaling"
    }
  ],
  "note_path": "docs/research-notes/energy-fitness-generator-literature-2026-06-23.md",
  "preconditions_checked": {
    "agents_md_read": true,
    "arxiv_http_200_verified_ids": [
      "https://arxiv.org/abs/1710.11089",
      "https://arxiv.org/abs/1810.04586",
      "https://arxiv.org/abs/2107.07031",
      "https://arxiv.org/abs/2302.04693",
      "https://arxiv.org/abs/2308.05483",
      "https://arxiv.org/abs/2502.02962",
      "https://arxiv.org/abs/2504.01915",
      "https://arxiv.org/abs/2504.04366",
      "https://arxiv.org/abs/2505.10819",
      "https://arxiv.org/abs/2506.07255",
      "https://arxiv.org/abs/2604.03208",
      "https://arxiv.org/abs/2605.05138",
      "https://arxiv.org/abs/2605.27130",
      "https://arxiv.org/abs/2605.28814"
    ],
    "codex_md_read": true,
    "deep_research_invoked": false,
    "energy_config_note_read": true,
    "exp4637_artifact_read": true,
    "leaderboard_submission": false,
    "live_llm_inference": false,
    "model_load": false,
    "network_hf_models_reachable": true,
    "ops_docs_modified": false,
    "research_conductor_modified": false,
    "research_references_read": true,
    "research_studying_read": true,
    "sweep_clusters_urls": [
      "http://export.arxiv.org/api/query?search_query=(abs:\"affordance\"+OR+abs:\"action+effect\"+OR+abs:\"clickability\"+OR+abs:\"frame+prediction\"+OR+abs:\"intrinsic+motivation\"+OR+abs:\"directed+exploration\"+OR+abs:\"novelty+search\")+AND+(abs:\"reinforcement+learning\"+OR+abs:\"agent\"+OR+abs:\"exploration\"+OR+abs:\"interactive+environment\"+OR+abs:\"ARC\")&start=0&max_results=8&sortBy=submittedDate&sortOrder=descending",
      "http://export.arxiv.org/api/query?search_query=(abs:\"neural+guided+search\"+OR+abs:\"learned+heuristic\"+OR+abs:\"value+guided+search\"+OR+abs:\"program+induction\"+OR+abs:\"world+model\"+OR+abs:\"goal+induction\")+AND+(abs:\"planning\"+OR+abs:\"agent\"+OR+abs:\"reasoning\"+OR+abs:\"reinforcement+learning\")&start=0&max_results=8&sortBy=submittedDate&sortOrder=descending"
    ],
    "sweep_clusters_used": true,
    "sweep_semscholar_arxiv_ids": [],
    "sweep_semscholar_queries": [
      "energy-as-fitness quality diversity MAP-Elites action sequences ARC generation 2605.28814",
      "FunSearch quality diversity program search action sequence ARC 2308.05483 2504.01915",
      "macro-action vocabulary empowerment eigenoptions affordance 2107.07031 2502.02962 2302.04693 1810.04586",
      "hierarchical subgoal MCTS ARC world model 2604.03208 2506.07255 2605.05138",
      "PoE-World product of experts executable world model ARC 2605.05138"
    ],
    "sweep_semscholar_rate_limited_queries": [
      "FunSearch quality diversity program search action sequence ARC 2308.05483 2504.01915",
      "macro-action vocabulary empowerment eigenoptions affordance 2107.07031 2502.02962 2302.04693 1810.04586",
      "hierarchical subgoal MCTS ARC world model 2604.03208 2506.07255 2605.05138",
      "PoE-World product of experts executable world model ARC 2605.05138"
    ],
    "sweep_semscholar_used": true,
    "training_launched": false,
    "verifier_gaps_read": true,
    "websearch_webfetch_top_sources": [
      "https://arxiv.org/abs/2605.28814",
      "https://arxiv.org/abs/2308.05483",
      "https://arxiv.org/abs/2504.01915",
      "https://arxiv.org/abs/2605.27130",
      "https://arxiv.org/abs/2107.07031",
      "https://arxiv.org/abs/2502.02962",
      "https://arxiv.org/abs/2302.04693",
      "https://arxiv.org/abs/1810.04586",
      "https://arxiv.org/abs/1710.11089",
      "https://arxiv.org/abs/2604.03208",
      "https://arxiv.org/abs/2506.07255",
      "https://arxiv.org/abs/2504.04366",
      "https://arxiv.org/abs/2505.10819",
      "https://arxiv.org/abs/2605.05138",
      "https://www.nature.com/articles/s41586-023-06924-6"
    ]
  },
  "random_seed": 4649
}
```

## Fresh-pass provenance

Read `AGENTS.md`, `CODEX.md`,
`results/experiment_4637_sota_ingestion_intrinsic_motivation.json`,
`docs/research-notes/arc-generation-wall-energy-config-space-2026-06-22.md`,
`ops/verifier_gaps.md`, `research-studying.md`, and
`research-references.md`. The filtered track was the .428 headline open
problem: ENERGY DRIVES GENERATION. The current generator is a goal-energy
heuristic plus action-effect expansion prior; this pass maps methods for the
next generator: energy-as-fitness QD evolution, macro-action vocabulary, and
hierarchical subgoal search feeding .429.

Reliable-channel pass, not `/deep-research`:
- `.venv/bin/python -c "import urllib.request; urllib.request.urlopen('https://huggingface.co/api/models', timeout=10); print('net_ok')"`
- `.venv/bin/python scripts/sweep_clusters.py 5 --max-results 8`
- `.venv/bin/python scripts/sweep_clusters.py 6 --max-results 8`
- `.venv/bin/python scripts/sweep_semscholar.py - --limit 8` with five focused queries
- low-concurrency WebSearch/WebFetch of the top arXiv papers plus FunSearch as non-arXiv comparator
- direct arXiv HTTP checks for all cited IDs

Semantic Scholar returned HTTP 429 for four focused queries and no S2-only
source was promoted. Direct arXiv HTTP checks returned 200 for arXiv:2605.28814,
arXiv:2308.05483, arXiv:2504.01915, arXiv:2605.27130, arXiv:2107.07031,
arXiv:2502.02962, arXiv:2302.04693, arXiv:1810.04586, arXiv:1710.11089,
arXiv:2604.03208, arXiv:2506.07255, arXiv:2504.04366, arXiv:2505.10819, and
arXiv:2605.05138. FunSearch / Nature s41586-023-06924-6 was treated only as a
non-arXiv evaluator-guided evolution comparator; every method claim below has
real arXiv source IDs. No live LLM inference, No training, No leaderboard submission,
no model load, and no live solve claim were run or made.
`scripts/research_conductor.py`, `ops/changelog.md`, and `ops/status.md` were
not edited by this workflow.

## SOTA -> experiment mapping

## Energy-as-fitness QD/BES action-sequence evolution

**Sources:** Bidirectional Evolutionary Search, arXiv:2605.28814; Quality
Diversity under Sparse Reward and Sparse Interaction, arXiv:2308.05483;
Unsupervised Quality-Diversity for deceptive fitness, arXiv:2504.01915.

**Mapping to goal-energy / action-effect:** the current search can rank
candidates after generation, but the winning action sequence is often absent.
BES supplies the recombination pattern: splice partial trajectories and add
backward subgoal feedback rather than relying on autoregressive expansion.
QD supplies the archive: keep elites across behavior niches such as objects
moved, max level reached, avatar region, and action-effect novelty. For .429,
goal-energy becomes the dense fitness and action-effect becomes the transition
feasibility model that filters invalid sequence variants.

**Implementation cost over current stack:** medium. Reuse existing
graph-explore trajectories, state hashes, goal-energy deltas, and action-effect
cell-recall; add sequence mutation, shared-state crossover, archive descriptors,
and budget-normalized first-win/action-count gates.

**Fails when:** goal-energy is wrong or too sparse, action-effect aliases hidden
state, descriptors reward visual novelty instead of progress, or the archive
spends the fixed interaction budget without increasing `winner_generated`.

## Macro-action vocabulary from empowerment, proto-goals, and eigenoptions

**Sources:** empowerment in sparse-reward exploration, arXiv:2107.07031;
intrinsic motivation as constrained entropy maximization, arXiv:2502.02962;
proto-goal pruning, arXiv:2302.04693; Laplacian representations,
arXiv:1810.04586; eigenoptions, arXiv:1710.11089.

**Mapping to goal-energy / action-effect:** the current `rich_action_candidates`
surface is still primitive-action-first. A macro-action vocabulary converts
reliable multi-step effects into one planner token: push-until-blocked,
cycle-color, toggle-then-step, and reach-a-proto-goal. Goal-energy decides which
macro endpoints matter; action-effect validates repeatable preconditions and
effects before a macro can enter the shared library.

**Implementation cost over current stack:** medium-high. Mine exploration
traces, segment them by effect, cluster by controllability/reachability, keep
empowerment-positive macros, and expose them to the candidate router behind a
primitive replay check.

**Fails when:** high-control loops have no goal value, per-game macros do not
transfer, online eigenoption/proto-goal discovery consumes the same budget it
tries to save, or macro abstraction hides the primitive transition that the
verifier must inspect.

## Hierarchical subgoal search over trusted goal-energy models

**Sources:** Hierarchical Planning with Latent World Models, arXiv:2604.03208;
Subgoal-Guided Policy Heuristic Search with Learned Subgoals, arXiv:2506.07255;
Sokoban hierarchical landmarks, arXiv:2504.04366; Executable World Models for
ARC-AGI-3, arXiv:2605.05138.

**Mapping to goal-energy / action-effect:** GAP-ARCH-NO-HIERARCHICAL-SEARCH is
the structural miss: a flat frontier cannot reliably cross long-horizon levels.
The .429 path is a high-level planner that proposes subgoals, with low-level
MCTS or best-first search verifying each subgoal through action-effect dynamics.
Goal-energy scores the subgoal and final state, while action-effect prevents
the hierarchy from planning through impossible transitions.

**Implementation cost over current stack:** medium. Add a subgoal layer over
graph exploration, mine failed search trees for useful intermediate states,
route each subgoal through bounded replay-verified low-level search, and retain
uniform-energy ablations.

**Fails when:** subgoals are visually plausible but not goal-relevant, learned
subgoals require complete solutions the current search lacks, MCTS expands
through an untrusted model, or hierarchy reduces node count while increasing
primitive actions.

## PoE-World factored executable model with plan-through-model search

**Sources:** PoE-World, arXiv:2505.10819; Executable World Models for ARC-AGI-3,
arXiv:2605.05138.

**Mapping to goal-energy / action-effect:** monolithic exact-grid induction
fails hard when one rule is wrong. PoE-World suggests a softer factored model:
small programmatic experts for object-level preconditions/effects, combined by
weights rather than a single all-or-nothing transition law. Goal-energy scores
expert-set simplicity and target satisfaction; action-effect supplies observed
transition rows to weight, prune, and replay-check experts before planning.

**Implementation cost over current stack:** medium-high. Need program-expert
induction, trust-weighted composition, held-out transition checks, and a planner
that emits only replay-verified candidate sequences.

**Fails when:** experts are not independent, rare object interactions are
smoothed away, generated code overfits prefix transitions, or soft planning
produces live-invalid actions.

## Distributed QD mutation ensemble as a later scaling lever

**Sources:** DEI distributed QD search, arXiv:2605.27130; BES,
arXiv:2605.28814.

**Mapping to goal-energy / action-effect:** this is not the first .429 build.
It becomes useful only after the single-node archive has signal. At that point,
heterogeneous mutation policies can diversify sequence edits while sharing the
same goal-energy fitness and action-effect replay verifier.

**Implementation cost over current stack:** high. Requires multiple mutation
policies, archive merge semantics, dedupe, replay verification, and strict
budget-normalized comparison against the single-node QD baseline.

**Fails when:** the base archive is empty of useful elites, diversity only adds
invalid or duplicate actions, merged elites are not replay-verified, or
distributed compute hides a worse per-budget generator.

## Bottom line for the .429 roadmap

1. Build `flagged_for_v429: energy_as_fitness_qd_bes_action_sequence_generator`
   first. BES arXiv:2605.28814 plus QD sparse/deceptive controls
   arXiv:2308.05483 and arXiv:2504.01915 are the direct answer to
   absent-winner generation.
2. Pair it with
   `flagged_for_v429: macro_action_vocabulary_empowerment_options`. The macro
   layer uses empowerment/proto-goal/eigenoption sources arXiv:2107.07031,
   arXiv:2502.02962, arXiv:2302.04693, arXiv:1810.04586, and arXiv:1710.11089
   to collapse horizon before QD burns budget.
3. Add
   `flagged_for_v429: hierarchical_subgoal_search_over_goal_energy` when the
   archive can already replay near-winners. HWM arXiv:2604.03208, subgoal-PHS
   arXiv:2506.07255, and Sokoban landmarks arXiv:2504.04366 make this the
   right fix for GAP-ARCH-NO-HIERARCHICAL-SEARCH.
4. Use
   `flagged_for_v429: poe_world_factored_executable_model_planner` as the
   factored model arm after replay checks are reliable. PoE-World
   arXiv:2505.10819 plus ARC executable world models arXiv:2605.05138 map cleanly
   onto the current verifier-as-model-trust stack.
5. Keep `flagged_for_v429: distributed_qd_mutation_ensemble_later_scaling`,
   arXiv:2605.27130 plus arXiv:2605.28814, as a scaling experiment after the
   single-node generator beats the current goal-energy/action-effect baseline.

