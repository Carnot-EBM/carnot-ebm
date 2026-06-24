# Generation-guidance SOTA ingestion 2026-06-24

```json
{
  "citations_verified": {
    "1011.0686": {
      "http_status": 200,
      "title": "A Reduction of Imitation Learning and Structured Prediction to No-Regret Online Learning",
      "url": "https://arxiv.org/abs/1011.0686"
    },
    "1706.04599": {
      "http_status": 200,
      "title": "On Calibration of Modern Neural Networks",
      "url": "https://arxiv.org/abs/1706.04599"
    },
    "2102.04518": {
      "http_status": 200,
      "title": "A* Search Without Expansions: Learning Heuristic Functions with Deep Q-Networks",
      "url": "https://arxiv.org/abs/2102.04518"
    },
    "2308.05483": {
      "http_status": 200,
      "title": "Quality Diversity under Sparse Reward and Sparse Interaction",
      "url": "https://arxiv.org/abs/2308.05483"
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
    "2604.11351": {
      "http_status": 200,
      "title": "WM-DAgger: Enabling Efficient Data Aggregation for Imitation Learning",
      "url": "https://arxiv.org/abs/2604.11351"
    },
    "2605.05138": {
      "http_status": 200,
      "title": "Executable World Models for ARC-AGI-3 in the Era of Coding Agents",
      "url": "https://arxiv.org/abs/2605.05138"
    },
    "2605.28814": {
      "http_status": 200,
      "title": "Self-Improving Language Models with Bidirectional Evolutionary Search",
      "url": "https://arxiv.org/abs/2605.28814"
    }
  },
  "dead_levers_not_reflagged": [
    "macro-action horizon-collapse RETIRED",
    "click-heatmap off-centroid generator RETIRED",
    "just-explore schedule-extraction CLOSED",
    "goal-energy heuristic NULL"
  ],
  "deep_research_not_used": true,
  "field_principles": {
    "citations_verified": {
      "principle": "each cited arXiv ID with an HTTP-200 verification -- pre-empts fabricated citations."
    },
    "dead_levers_not_reflagged": {
      "principle": "names the DEAD levers (macro/click/schedule/goal-energy heuristic) confirmed NOT re-flagged -- honors the week's falsifications."
    },
    "deep_research_not_used": {
      "principle": "MUST be true -- /deep-research is BANNED in the autonomous loop; used sweep helpers + low-concurrency WebSearch/WebFetch."
    },
    "flagged_for_next_roadmap": {
      "principle": "the strongest method(s) flagged as candidate .430 inputs (flagged_for_v430) -- closes discover->ingest->plan->experiment."
    },
    "honest_verdict": {
      "principle": "terminal prefix; success: sota_ingestion_generation_guidance_mapped."
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
    "flagged_for_v430: hierarchical_subgoal_e3_frontier_with_distribution_shift_value_routing (arXiv:2604.03208 + arXiv:2506.07255 + arXiv:2504.04366 + arXiv:1011.0686 + arXiv:2604.11351 + arXiv:1706.04599)",
    "flagged_for_v430: poe_world_factored_executable_subgoal_planner (arXiv:2505.10819 + arXiv:2605.05138)"
  ],
  "honest_verdict": "success: sota_ingestion_generation_guidance_mapped",
  "inference_substrate": "aggregation_from_upstream_artifacts",
  "methods_mapped": [
    {
      "fails_when": "the subgoal miner proposes visual states that are not goal relevant, value routing is still shifted from live frontier states, QD spends budget around an unreachable landmark, or live E3 replay rejects the subgoal path.",
      "implement_cost_over_current_stack": "medium: add a high-level subgoal layer above the current live E3 frontier, mine failed A1/A2 search trees for subgoal candidates, route each subgoal through bounded low-level search, and keep replay checks plus matched no-regression controls.",
      "maps_to_current_stack": "A1 value-routing supplies a calibrated low-cost tie-breaker inside each subgoal search, A2 energy-fitness QD becomes a subgoal-conditioned sequence proposer rather than a standalone archive, and live E3 remains the replay-verified executor and parity surface.",
      "method": "Hierarchical subgoal search over the live E3 frontier",
      "roadmap_candidate": "flagged_for_v430: hierarchical_subgoal_e3_frontier_with_distribution_shift_value_routing (arXiv:2604.03208 + arXiv:2506.07255 + arXiv:2504.04366 + arXiv:1011.0686 + arXiv:2604.11351 + arXiv:1706.04599)",
      "source_ids": [
        "2604.03208",
        "2506.07255",
        "2504.04366",
        "2605.05138"
      ],
      "track": "hierarchical_subgoal_search_live_e3_frontier"
    },
    {
      "fails_when": "expert factors are not independent, rare interactions are smoothed away by the product, generated experts overfit prefix transitions, or the soft plan yields actions that live E3 cannot replay.",
      "implement_cost_over_current_stack": "medium-high: induce small programmatic experts for object-level preconditions and effects, weight them by held-out transition trust, compose only replay-stable factors, and plan subgoal-conditioned candidate sequences through the product model.",
      "maps_to_current_stack": "A1 value-routing scores which expert-predicted states deserve live expansion, A2 energy-fitness QD mutates only sequences that the factored executable model marks feasible, and live E3 adjudicates every emitted plan through its normal action/replay path.",
      "method": "PoE-World factored executable model subgoal planner",
      "roadmap_candidate": "flagged_for_v430: poe_world_factored_executable_subgoal_planner (arXiv:2505.10819 + arXiv:2605.05138)",
      "source_ids": [
        "2505.10819",
        "2605.05138"
      ],
      "track": "poe_world_factored_executable_model_planner"
    },
    {
      "fails_when": "the aggregated frontier data is too small, calibration collapses under hidden-state games, cached Q-values become stale after level-up, or the router only reorders candidates that A2 and live E3 still fail to generate.",
      "implement_cost_over_current_stack": "medium: collect live-frontier states from A1 and A2 failures, use DAgger-style aggregation to retrain or recalibrate the value router on off-path states, convert scores to bounded cost deltas, and cache decision-point evaluations so routing stays affordable.",
      "maps_to_current_stack": "A1 value-routing stops applying a winning-path value head to shifted live frontier states, A2 energy-fitness QD receives calibrated subgoal costs instead of raw goal-energy ranking, and live E3 keeps primitive actions plus parity gates as the scored integration point.",
      "method": "Distribution-shift-corrected value routing for subgoal frontiers",
      "roadmap_candidate": "flagged_for_v430: hierarchical_subgoal_e3_frontier_with_distribution_shift_value_routing (arXiv:2604.03208 + arXiv:2506.07255 + arXiv:2504.04366 + arXiv:1011.0686 + arXiv:2604.11351 + arXiv:1706.04599)",
      "source_ids": [
        "1011.0686",
        "2604.11351",
        "1706.04599",
        "2102.04518"
      ],
      "track": "distribution_shift_corrected_value_routing"
    }
  ],
  "note_path": "docs/research-notes/generation-guidance-sota-ingestion-2026-06-24.md",
  "preconditions_checked": {
    "a1_value_routing_artifact_read": true,
    "a2_energy_fitness_qd_artifact_read": true,
    "agents_md_read": true,
    "arxiv_http_200_verified_ids": [
      "https://arxiv.org/abs/1011.0686",
      "https://arxiv.org/abs/1706.04599",
      "https://arxiv.org/abs/2102.04518",
      "https://arxiv.org/abs/2308.05483",
      "https://arxiv.org/abs/2504.01915",
      "https://arxiv.org/abs/2504.04366",
      "https://arxiv.org/abs/2505.10819",
      "https://arxiv.org/abs/2506.07255",
      "https://arxiv.org/abs/2604.03208",
      "https://arxiv.org/abs/2604.11351",
      "https://arxiv.org/abs/2605.05138",
      "https://arxiv.org/abs/2605.28814"
    ],
    "codex_md_read": true,
    "dead_lever_notes_read": [
      "macro-action horizon-collapse RETIRED",
      "click-heatmap off-centroid generator RETIRED",
      "just-explore schedule-extraction CLOSED",
      "goal-energy heuristic NULL"
    ],
    "deep_research_invoked": false,
    "exp4649_artifact_read": true,
    "exp4649_note_read": true,
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
      "http://export.arxiv.org/api/query?search_query=(abs:\"affordance\"+OR+abs:\"action+effect\"+OR+abs:\"clickability\"+OR+abs:\"frame+prediction\"+OR+abs:\"intrinsic+motivation\"+OR+abs:\"directed+exploration\"+OR+abs:\"novelty+search\")+AND+(abs:\"reinforcement+learning\"+OR+abs:\"agent\"+OR+abs:\"exploration\"+OR+abs:\"interactive+environment\"+OR+abs:\"ARC\")&start=0&max_results=8&sortBy=submittedDate&sortOrder=descending",
      "http://export.arxiv.org/api/query?search_query=(abs:\"neural+guided+search\"+OR+abs:\"learned+heuristic\"+OR+abs:\"value+guided+search\"+OR+abs:\"program+induction\"+OR+abs:\"world+model\"+OR+abs:\"goal+induction\")+AND+(abs:\"planning\"+OR+abs:\"agent\"+OR+abs:\"reasoning\"+OR+abs:\"reinforcement+learning\")&start=0&max_results=8&sortBy=submittedDate&sortOrder=descending"
    ],
    "sweep_clusters_used": true,
    "sweep_semscholar_arxiv_ids": [],
    "sweep_semscholar_queries": [
      "hierarchical subgoal search learned subgoals latent world model ARC 2604.03208 2506.07255 2504.04366",
      "factored executable world model product of experts programmatic experts ARC 2505.10819 2605.05138",
      "distribution shift corrected value routing affordable value routing offline to live 1011.0686 2604.11351 1706.04599 2102.04518"
    ],
    "sweep_semscholar_rate_limited_queries": [
      "hierarchical subgoal search learned subgoals latent world model ARC 2604.03208 2506.07255 2504.04366",
      "factored executable world model product of experts programmatic experts ARC 2505.10819 2605.05138",
      "distribution shift corrected value routing affordable value routing offline to live 1011.0686 2604.11351 1706.04599 2102.04518"
    ],
    "sweep_semscholar_used": true,
    "training_launched": false,
    "websearch_webfetch_top_sources": [
      "https://arxiv.org/abs/2604.03208",
      "https://arxiv.org/abs/2506.07255",
      "https://arxiv.org/abs/2504.04366",
      "https://arxiv.org/abs/2505.10819",
      "https://arxiv.org/abs/2605.05138",
      "https://arxiv.org/abs/1011.0686",
      "https://arxiv.org/abs/2604.11351",
      "https://arxiv.org/abs/1706.04599"
    ]
  },
  "random_seed": 4661
}
```

## Fresh-pass provenance

Read `AGENTS.md`, `CODEX.md`,
`results/experiment_4649_sota_ingestion_energy_fitness_generator.json`,
`docs/research-notes/energy-fitness-generator-literature-2026-06-23.md`,
`research-studying.md`, `research-references.md`,
`docs/research-notes/macro-vocab-prototype-finding-2026-06-23.md`,
`docs/research-notes/click-heatmap-generator-falsified-2026-06-23.md`,
`docs/research-notes/h2h-just-explore-vs-bare-explorer-2026-06-23.md`,
`results/experiment_4652_value_routing_cost_fix_live.json`, and
`results/experiment_4653_energy_fitness_qd_generation_live.json`. The current
stack is A1 value-routing plus A2 energy-fitness QD inside the live E3 agent
path; both returned no live lift, so this pass maps surviving SOTA directions
that can add generation guidance rather than re-ranking the same empty pool.

Reliable-channel pass, not `/deep-research`:
- `curl -sf -o /dev/null https://huggingface.co/api/models`
- `.venv/bin/python scripts/sweep_clusters.py --help`
- `.venv/bin/python scripts/sweep_clusters.py 5 --max-results 8`
- `.venv/bin/python scripts/sweep_clusters.py 6 --max-results 8`
- `.venv/bin/python scripts/sweep_semscholar.py - --limit 8` with three focused queries
- low-concurrency WebSearch/WebFetch of the top hierarchical, factored-model, and value-routing papers
- direct arXiv HTTP checks for all cited IDs

Semantic Scholar returned HTTP 429 for the focused queries and no S2-only source
was promoted. Direct arXiv HTTP checks returned 200 for arXiv:2604.03208,
arXiv:2506.07255, arXiv:2504.04366, arXiv:2505.10819, arXiv:2605.05138,
arXiv:1011.0686, arXiv:2604.11351, arXiv:1706.04599, arXiv:2102.04518,
arXiv:2605.28814, arXiv:2308.05483, and arXiv:2504.01915. The A2 current-stack
context is BES/QD action-sequence evolution from arXiv:2605.28814,
arXiv:2308.05483, and arXiv:2504.01915, but the live artifact did not generate
a winner. No live LLM inference, No training, No leaderboard submission, no
model load, and no live solve claim were run or made. `scripts/research_conductor.py`,
`ops/changelog.md`, and `ops/status.md` were not edited by this workflow.

Dead levers confirmed not re-flagged: macro-action horizon-collapse RETIRED;
click-heatmap off-centroid generator RETIRED; just-explore schedule-extraction
CLOSED; goal-energy heuristic NULL.

## SOTA -> experiment mapping

## Hierarchical subgoal search over the live E3 frontier

**Sources:** Hierarchical Planning with Latent World Models, arXiv:2604.03208;
Subgoal-Guided Policy Heuristic Search with Learned Subgoals, arXiv:2506.07255;
Sokoban hierarchical landmarks, arXiv:2504.04366; Executable World Models for
ARC-AGI-3, arXiv:2605.05138.

**Mapping to current stack:** A1 value-routing should stop acting as a global
ranker and instead serve as a calibrated tie-breaker inside each low-level
subgoal search. A2 energy-fitness QD should mutate sequences under a named
subgoal rather than evolve a broad standalone archive. The live E3 path remains
the only scored executor: every candidate path must replay under the existing
action and parity gates.

**Implementation cost over current stack:** medium. Add subgoal mining from
failed A1/A2 trees, run bounded low-level search for each selected subgoal, and
retain matched baseline, random/subgoal ablation, and replay gates.

**Fails when:** mined subgoals are only visually plausible, A1 remains
distribution-shifted on live frontier states, A2 burns budget near unreachable
landmarks, or live E3 rejects the replayed path.

## PoE-World factored executable model subgoal planner

**Sources:** PoE-World, arXiv:2505.10819; Executable World Models for
ARC-AGI-3, arXiv:2605.05138.

**Mapping to current stack:** A1 value-routing scores which expert-predicted
states deserve expansion; A2 energy-fitness QD mutates only sequences that the
factored model says are feasible; live E3 executes and audits the emitted plans.
This directly attacks the bridge gap left by the A2 null: QD needs a better
feasibility model before it mutates.

**Implementation cost over current stack:** medium-high. Induce object-level
precondition/effect experts, weight them by held-out transition trust, compose
only replay-stable experts, and search through the product model with hard live
replay checks.

**Fails when:** expert factors are not independent, rare object interactions
are smoothed away, generated experts overfit prefix transitions, or soft model
planning emits a live-invalid action sequence.

## Distribution-shift-corrected value routing for subgoal frontiers

**Sources:** DAgger, arXiv:1011.0686; WM-DAgger, arXiv:2604.11351; calibration,
arXiv:1706.04599; A* value heuristics, arXiv:2102.04518.

**Mapping to current stack:** A1 value-routing failed as a cost-fixed live lift,
which points to residual distribution shift or calibration. The repair is not a
bigger raw value weight. Aggregate the live frontier states where A1/A2 fail,
calibrate scores into bounded costs, and cache decision-point Q/value estimates.
A2 energy-fitness QD then receives subgoal costs instead of raw goal-energy
ranking, while live E3 keeps primitive actions and parity gates.

**Implementation cost over current stack:** medium. Add frontier-state logging,
DAgger-style aggregation or WM-DAgger rollouts, temperature/isotonic calibration
for cost deltas, and decision-point caching under the existing affordability
guard.

**Fails when:** aggregated frontier data is too small, hidden-state games break
calibration, cached values go stale after level-up, or routing only reorders a
candidate pool that still lacks the winning action.

## Bottom line for the .430 roadmap

1. Build `flagged_for_v430: hierarchical_subgoal_e3_frontier_with_distribution_shift_value_routing`
   first. It combines the strongest search structure, arXiv:2604.03208,
   arXiv:2506.07255, and arXiv:2504.04366, with the value-distribution repair
   from arXiv:1011.0686, arXiv:2604.11351, and arXiv:1706.04599.
2. Keep `flagged_for_v430: poe_world_factored_executable_subgoal_planner` as the
   second live level-up candidate. PoE-World arXiv:2505.10819 plus executable
   ARC world models arXiv:2605.05138 is the best factored-model answer to A2's
   no-winner bridge gap.
3. Treat raw BES/QD sources arXiv:2605.28814, arXiv:2308.05483, and
   arXiv:2504.01915 as current-stack context only until subgoal or factored-model
   guidance changes the candidate pool.
4. Do not re-open macro-action horizon-collapse RETIRED, click-heatmap
   off-centroid generator RETIRED, just-explore schedule-extraction CLOSED, or
   goal-energy heuristic NULL as `.430` inputs.

