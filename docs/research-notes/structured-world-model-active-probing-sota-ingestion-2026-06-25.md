# Structured-world-model and active-probing SOTA ingestion 2026-06-25

```json
{
  "citations_verified": {
    "2210.13455": {
      "http_status": 200,
      "title": "Epistemic Monte Carlo Tree Search",
      "url": "https://arxiv.org/abs/2210.13455"
    },
    "2307.02427": {
      "http_status": 200,
      "title": "FOCUS: Object-Centric World Models for Robotics Manipulation",
      "url": "https://arxiv.org/abs/2307.02427"
    },
    "2309.08477": {
      "http_status": 200,
      "title": "Deep Multi-Agent Reinforcement Learning for Decentralized Active Hypothesis Testing",
      "url": "https://arxiv.org/abs/2309.08477"
    },
    "2410.08822": {
      "http_status": 200,
      "title": "SOLD: Slot Object-Centric Latent Dynamics Models for Relational Manipulation Learning from Pixels",
      "url": "https://arxiv.org/abs/2410.08822"
    },
    "2506.01876": {
      "http_status": 200,
      "title": "In-Context Learning for Pure Exploration",
      "url": "https://arxiv.org/abs/2506.01876"
    },
    "2511.02225": {
      "http_status": 200,
      "title": "Learning Interactive World Model for Object-Centric Reinforcement Learning",
      "url": "https://arxiv.org/abs/2511.02225"
    },
    "2511.06136": {
      "http_status": 200,
      "title": "When Object-Centric World Models Meet Policy Learning: From Pixels to Policies, and Where It Breaks",
      "url": "https://arxiv.org/abs/2511.06136"
    },
    "2601.06604": {
      "http_status": 200,
      "title": "Object-Centric World Models Meet Monte Carlo Tree Search",
      "url": "https://arxiv.org/abs/2601.06604"
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
      "principle": "the strongest method(s) flagged as candidate .434 inputs (flagged_for_v434) -- closes discover->ingest->plan->experiment."
    },
    "honest_verdict": {
      "principle": "terminal prefix; success: sota_ingestion_structured_world_model_mapped."
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
    "flagged_for_v434: factored_object_relational_executable_world_model (arXiv:2511.02225 + arXiv:2410.08822 + arXiv:2307.02427)",
    "flagged_for_v434: object_model_mcts_with_epistemic_probe_planning (arXiv:2601.06604 + arXiv:2210.13455)",
    "flagged_for_v434: hypothesis_driven_active_probe_loop (arXiv:2506.01876 + arXiv:2309.08477)"
  ],
  "honest_verdict": "success: sota_ingestion_structured_world_model_mapped",
  "inference_substrate": "aggregation_from_upstream_artifacts",
  "methods_mapped": [
    {
      "fails_when": "object slots drift under off-path interactions, interaction factors alias hidden registers, or the trusted-factor ledger overfits short public prefixes and produces plans that fail on live transitions.",
      "implement_cost_over_current_stack": "high: lift A1 object slots and relations into typed transition factors, extend arc_executable_world_model beyond full-grid exact matching into held-out object/interaction trust, and let A2 traces seed factor induction rather than only ranking first-contact actions.",
      "maps_to_current_stack": "live E3 explorer uses the induced factors as its planning substrate; arc_executable_world_model becomes a product of object and interaction rules instead of one monolithic grid engine; A1 object-centric perception supplies slots and relations; A2 amortized prior plus Go-Explore supplies replayable prefixes and action-effect evidence.",
      "method": "Factored object-relational executable transition model",
      "residual_scope": "A1 residual: object_centric_perception_no_new_level_residual_offpath_calibration_insufficient, with deployable object-centric coverage still not banking a level; A2 residual: amortized_prior_go_explore_no_coverage_gain_residual_logged, with candidate_generation_coverage_with_prior equal to the no-prior baseline. The scoped fallback is the structured-world-model / active-probing next wall: induce an executable object-relational transition model at runtime, plan inside it, and run targeted probes that confirm or refute explicit mechanic hypotheses before spending more live actions.",
      "roadmap_candidate": "flagged_for_v434: factored_object_relational_executable_world_model (arXiv:2511.02225 + arXiv:2410.08822 + arXiv:2307.02427)",
      "source_ids": [
        "2511.02225",
        "2410.08822",
        "2307.02427"
      ],
      "track": "factored_object_relational_executable_world_model"
    },
    {
      "fails_when": "uncertainty is uncalibrated, model errors compound over long rollouts, the branching factor remains grid-scale rather than object-scale, or live action budgets cannot afford enough confirmation probes.",
      "implement_cost_over_current_stack": "medium-high: replace the current bounded BFS plan_in_model fallback with an MCTS planner over the object-relational model, propagate model uncertainty through rollouts, and allocate live probes to high-value uncertain branches before executing a candidate solution prefix.",
      "maps_to_current_stack": "live E3 explorer asks MCTS for both solve actions and probe actions; arc_executable_world_model supplies the rollout engine and trust weights; A1 object-centric perception defines object graph states; A2 amortized prior plus Go-Explore returns to archived cells before testing uncertain branches.",
      "method": "Object-model MCTS with epistemic probe planning",
      "residual_scope": "A1 residual: object_centric_perception_no_new_level_residual_offpath_calibration_insufficient, with deployable object-centric coverage still not banking a level; A2 residual: amortized_prior_go_explore_no_coverage_gain_residual_logged, with candidate_generation_coverage_with_prior equal to the no-prior baseline. The scoped fallback is the structured-world-model / active-probing next wall: induce an executable object-relational transition model at runtime, plan inside it, and run targeted probes that confirm or refute explicit mechanic hypotheses before spending more live actions.",
      "roadmap_candidate": "flagged_for_v434: object_model_mcts_with_epistemic_probe_planning (arXiv:2601.06604 + arXiv:2210.13455)",
      "source_ids": [
        "2601.06604",
        "2210.13455"
      ],
      "track": "object_model_mcts_with_epistemic_probe_planning"
    },
    {
      "fails_when": "the hypothesis class omits the true mechanic, probe outcomes are not distinguishable at logical-grid resolution, or the agent spends its action budget identifying a rule that is not sufficient for level completion.",
      "implement_cost_over_current_stack": "medium: add an explicit mechanic-hypothesis table, synthesize discriminating probe actions from the current object model, update posterior support after each observed transition, and expose stop/continue decisions to the live E3 explorer before it commits to a solve plan.",
      "maps_to_current_stack": "live E3 explorer alternates perceive -> hypothesize -> test -> refine; arc_executable_world_model predicts each hypothesis' transition outcome; A1 object-centric perception grounds the hypothesis predicates; A2 amortized prior plus Go-Explore supplies candidate probes and replayable reset points.",
      "method": "Hypothesis-driven active probe loop",
      "residual_scope": "A1 residual: object_centric_perception_no_new_level_residual_offpath_calibration_insufficient, with deployable object-centric coverage still not banking a level; A2 residual: amortized_prior_go_explore_no_coverage_gain_residual_logged, with candidate_generation_coverage_with_prior equal to the no-prior baseline. The scoped fallback is the structured-world-model / active-probing next wall: induce an executable object-relational transition model at runtime, plan inside it, and run targeted probes that confirm or refute explicit mechanic hypotheses before spending more live actions.",
      "roadmap_candidate": "flagged_for_v434: hypothesis_driven_active_probe_loop (arXiv:2506.01876 + arXiv:2309.08477)",
      "source_ids": [
        "2506.01876",
        "2309.08477"
      ],
      "track": "hypothesis_driven_active_probe_loop"
    },
    {
      "fails_when": "the drift metric is too conservative and rejects every useful induced model, or too permissive and lets visually plausible but causally wrong object rollouts pass into execution.",
      "implement_cost_over_current_stack": "low-medium: add held-out off-path drift diagnostics, per-factor rejection reasons, and plan invalidation when object latents or relations shift under multi-object interactions.",
      "maps_to_current_stack": "live E3 explorer refuses brittle plans when drift rises; arc_executable_world_model records why factors were rejected; A1 object-centric perception supplies latent stability checks; A2 amortized prior plus Go-Explore collects the off-path transitions that expose breakage.",
      "method": "Latent-drift and policy-breakage guardrails for object world models",
      "residual_scope": "A1 residual: object_centric_perception_no_new_level_residual_offpath_calibration_insufficient, with deployable object-centric coverage still not banking a level; A2 residual: amortized_prior_go_explore_no_coverage_gain_residual_logged, with candidate_generation_coverage_with_prior equal to the no-prior baseline. The scoped fallback is the structured-world-model / active-probing next wall: induce an executable object-relational transition model at runtime, plan inside it, and run targeted probes that confirm or refute explicit mechanic hypotheses before spending more live actions.",
      "roadmap_candidate": "guardrail_for_v434: prevent object-model planning false positives",
      "source_ids": [
        "2511.06136"
      ],
      "track": "object_world_model_policy_breakage_guardrails"
    }
  ],
  "note_path": "docs/research-notes/structured-world-model-active-probing-sota-ingestion-2026-06-25.md",
  "preconditions_checked": {
    "agents_md_read": true,
    "arc_executable_world_model_read": true,
    "arxiv_http_200_verified_ids": [
      "https://arxiv.org/abs/2210.13455",
      "https://arxiv.org/abs/2307.02427",
      "https://arxiv.org/abs/2309.08477",
      "https://arxiv.org/abs/2410.08822",
      "https://arxiv.org/abs/2506.01876",
      "https://arxiv.org/abs/2511.02225",
      "https://arxiv.org/abs/2511.06136",
      "https://arxiv.org/abs/2601.06604"
    ],
    "codex_md_read": true,
    "deep_research_invoked": false,
    "exp4697_artifact_read": true,
    "exp4697_note_read": true,
    "exp4700_artifact_read": true,
    "exp4701_artifact_read": true,
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
      "http://export.arxiv.org/api/query?search_query=(abs:\"affordance\"+OR+abs:\"action+effect\"+OR+abs:\"clickability\"+OR+abs:\"frame+prediction\"+OR+abs:\"intrinsic+motivation\"+OR+abs:\"directed+exploration\"+OR+abs:\"novelty+search\")+AND+(abs:\"reinforcement+learning\"+OR+abs:\"agent\"+OR+abs:\"exploration\"+OR+abs:\"interactive+environment\"+OR+abs:\"ARC\")&start=0&max_results=8&sortBy=submittedDate&sortOrder=descending",
      "http://export.arxiv.org/api/query?search_query=(abs:\"active+inference\"+OR+abs:\"free+energy\"+OR+abs:\"free+energy+principle\"+OR+abs:\"predictive+coding\"+OR+abs:\"world+model\")+AND+(abs:\"LLM\"+OR+abs:\"language+model\"+OR+abs:\"reasoning\")&start=0&max_results=8&sortBy=submittedDate&sortOrder=descending"
    ],
    "sweep_clusters_used": true,
    "sweep_semscholar_arxiv_ids": [
      "2503.06170",
      "2401.08577",
      "2601.06604",
      "2606.08775",
      "2408.11816",
      "2511.02225",
      "2502.07600",
      "2508.19828",
      "2309.08477"
    ],
    "sweep_semscholar_queries": [
      "object-centric world model interactive agent transition model planning",
      "hypothesis driven active probing active learning reinforcement learning agents"
    ],
    "sweep_semscholar_used": true,
    "training_launched": false,
    "websearch_webfetch_top_sources": [
      "https://arxiv.org/abs/2410.08822",
      "https://arxiv.org/abs/2511.02225",
      "https://arxiv.org/abs/2601.06604",
      "https://arxiv.org/abs/2511.06136",
      "https://arxiv.org/abs/2307.02427",
      "https://arxiv.org/abs/2210.13455",
      "https://arxiv.org/abs/2506.01876",
      "https://arxiv.org/abs/2309.08477"
    ]
  },
  "random_seed": 4709
}
```

## Fresh-pass provenance

Read `AGENTS.md`, `CODEX.md`,
`results/experiment_4697_sota_ingestion_amortized_exploration.json`,
`docs/research-notes/amortized-exploration-sota-ingestion-2026-06-24.md`,
`results/experiment_4700_object_centric_perception_proposal_live.json`,
`results/experiment_4701_amortized_exploration_prior_go_explore_live.json`,
`python/carnot/agentic/arc_executable_world_model.py`, `research-studying.md`,
and `research-references.md`. A1 closed with `object_centric_perception_no_new_level_residual_offpath_calibration_insufficient`: deployable
object-centric proposal coverage improved, but no live new level was banked.
A2 closed with `amortized_prior_go_explore_no_coverage_gain_residual_logged`: the amortized prior plus Go-Explore archive did
not raise candidate-generation coverage over the no-prior baseline. The .434
scope is therefore the structured-world-model / active-probing next wall: if perception and amortized exploration do
not surface the winning prefix, induce a structured executable transition model
and make the explorer plan and probe inside it.

Reliable-channel pass, not `/deep-research`:
- `curl -sf -o /dev/null https://huggingface.co/api/models`
- `.venv/bin/python scripts/sweep_clusters.py --help`
- `.venv/bin/python scripts/sweep_clusters.py 6 --max-results 8`
- `.venv/bin/python scripts/sweep_clusters.py 5 --max-results 8`
- `.venv/bin/python scripts/sweep_clusters.py 3 --max-results 8`
- `.venv/bin/python scripts/sweep_semscholar.py "object-centric world model interactive agent transition model planning" --limit 8`
- `.venv/bin/python scripts/sweep_semscholar.py "hypothesis driven active probing active learning reinforcement learning agents" --limit 8`
- low-concurrency WebSearch/WebFetch of the top structured-world-model and active-probing papers
- direct arXiv HTTP checks for all cited IDs

Direct arXiv HTTP checks returned 200 for arXiv:2410.08822, arXiv:2511.02225,
arXiv:2601.06604, arXiv:2511.06136, arXiv:2307.02427, arXiv:2210.13455,
arXiv:2506.01876, and arXiv:2309.08477. No live LLM inference, no training,
no leaderboard submission, no model load, and no live solve claim were run or
made. `scripts/research_conductor.py`, `ops/changelog.md`, and `ops/status.md`
were not edited by this workflow.

## SOTA -> .434 structured-world-model mapping

## Factored object-relational executable transition model

**Sources:** FIOC-WM, arXiv:2511.02225; SOLD, arXiv:2410.08822; FOCUS,
arXiv:2307.02427.

**Mapping to current stack:** convert A1's connected components, relation
keypoints, and object slots into typed transition factors. Extend
`arc_executable_world_model` from monolithic grid engines and exact-match
verification into a held-out trust ledger over object and interaction effects.
A1 object-centric perception supplies the representation substrate.
A2 amortized prior plus Go-Explore supplies replayable prefixes and action-effect
observations for factor induction.

**Implementation cost over current stack:** high. It requires a new factor
schema, held-out interaction scoring, and a planner that composes trusted
factors without assuming full-grid prediction is perfect.

**Fails when:** object slots drift under off-path interactions, interaction
factors alias hidden registers, or short prefixes overfit public-game mechanics.

## Object-model MCTS with epistemic probe planning

**Sources:** ObjectZero, arXiv:2601.06604; Epistemic MCTS, arXiv:2210.13455.

**Mapping to current stack:** replace the current bounded BFS-only
`plan_in_model` fallback with MCTS over the induced object model. The live E3
explorer asks the planner for both solution actions and probe actions, while
Go-Explore returns to archived states before testing uncertain branches.

**Implementation cost over current stack:** medium-high. The product world
model already exposes an executable engine, but MCTS needs state keys, rollout
budgets, uncertainty propagation, and a live-action policy for when to probe
versus execute.

**Fails when:** uncertainty is uncalibrated, model errors compound over rollout
depth, or the object abstraction fails to reduce the branching factor enough.

## Hypothesis-driven active probe loop

**Sources:** In-Context Pure Explorer, arXiv:2506.01876; MARLA for active
hypothesis testing, arXiv:2309.08477.

**Mapping to current stack:** make the agent maintain explicit hypotheses such
as "clicking a same-color object rewrites a target relation" or "the HUD count
gates level completion." `arc_executable_world_model` predicts outcomes under
each hypothesis, A1 grounds predicates in object slots, and A2/Go-Explore
provide candidate probes and reset points.

**Implementation cost over current stack:** medium. The current live E3 loop
already observes transitions; the missing piece is the explicit
perceive -> hypothesize -> test -> refine table plus targeted probe selection.

**Fails when:** the true mechanic is outside the hypothesis class, probe
outcomes are visually indistinguishable, or the action budget is spent
identifying a rule that is not sufficient to complete the level.

## Latent-drift and policy-breakage guardrails

**Source:** When Object-Centric World Models Meet Policy Learning, arXiv:2511.06136.

**Mapping to current stack:** add held-out off-path drift diagnostics and plan
invalidation so object-centric perception cannot create a false confidence
signal. The live E3 explorer refuses brittle object-model plans when A1 slots
or relations shift under multi-object interactions; `arc_executable_world_model`
records rejected factors and A2/Go-Explore collects the transitions that expose
breakage.

**Implementation cost over current stack:** low-medium. The ledger can be added
beside the existing rejected-factor diagnostics and verifier mismatch artifacts.

**Fails when:** the drift metric rejects every useful induced model or permits
visually plausible but causally wrong rollouts.

## Bottom line for the .434 roadmap

The strongest .434 input is
flagged_for_v434: factored_object_relational_executable_world_model
(arXiv:2511.02225 + arXiv:2410.08822 + arXiv:2307.02427). It attacks the next
wall directly by giving the explorer a structured executable transition model
instead of another proposal prior.

The planning companion is
flagged_for_v434: object_model_mcts_with_epistemic_probe_planning
(arXiv:2601.06604 + arXiv:2210.13455), and the active-learning companion is
flagged_for_v434: hypothesis_driven_active_probe_loop
(arXiv:2506.01876 + arXiv:2309.08477). Together they make the explorer choose
live actions that either solve in the induced model or maximally reduce
uncertainty about the game's mechanic.
