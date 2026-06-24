# Directed-exploration SOTA ingestion 2026-06-24

```json
{
  "citations_verified": {
    "1712.06560": {
      "http_status": 200,
      "title": "Improving Exploration in Evolution Strategies for Deep Reinforcement Learning via a Population of Novelty-Seeking Agents",
      "url": "https://arxiv.org/abs/1712.06560"
    },
    "1810.12894": {
      "http_status": 200,
      "title": "Exploration by Random Network Distillation",
      "url": "https://arxiv.org/abs/1810.12894"
    },
    "2002.06038": {
      "http_status": 200,
      "title": "Never Give Up: Learning Directed Exploration Strategies",
      "url": "https://arxiv.org/abs/2002.06038"
    },
    "2005.05960": {
      "http_status": 200,
      "title": "Planning to Explore via Self-Supervised World Models",
      "url": "https://arxiv.org/abs/2005.05960"
    },
    "2102.11137": {
      "http_status": 200,
      "title": "Program Synthesis Guided Reinforcement Learning for Partially Observed Environments",
      "url": "https://arxiv.org/abs/2102.11137"
    },
    "2502.10077": {
      "http_status": 200,
      "title": "Towards Empowerment Gain through Causal Structure Learning in Model-Based RL",
      "url": "https://arxiv.org/abs/2502.10077"
    },
    "2505.10819": {
      "http_status": 200,
      "title": "PoE-World: Compositional World Modeling with Products of Programmatic Experts",
      "url": "https://arxiv.org/abs/2505.10819"
    },
    "2603.02045": {
      "http_status": 200,
      "title": "Expanding LLM Agent Boundaries with Strategy-Guided Exploration",
      "url": "https://arxiv.org/abs/2603.02045"
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
      "principle": "the strongest method(s) flagged as candidate .432 inputs (flagged_for_v432) -- closes discover->ingest->plan->experiment."
    },
    "honest_verdict": {
      "principle": "terminal prefix; success: sota_ingestion_directed_exploration_mapped."
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
    "flagged_for_v432: controllable_novelty_e3_proposal_policy (arXiv:2002.06038 + arXiv:1810.12894 + arXiv:2603.02045)",
    "flagged_for_v432: program_synthesis_action_effect_proposal_filter (arXiv:2505.10819 + arXiv:2102.11137)"
  ],
  "honest_verdict": "success: sota_ingestion_directed_exploration_mapped",
  "inference_substrate": "aggregation_from_upstream_artifacts",
  "methods_mapped": [
    {
      "fails_when": "the embedding treats cosmetic frame changes as controllable novelty, the kNN table aliases distinct mechanics, novelty repeatedly revisits non-winning states, or strategy diversity generates language plans that do not ground to valid ARC actions.",
      "implement_cost_over_current_stack": "medium: add a state/action embedding over visible frame deltas and action-effect features, keep an episodic kNN novelty table plus an RND-style long-horizon novelty score, and run several exploration temperatures inside the same live E3 budget.",
      "maps_to_current_stack": "live E3 explorer receives a controllable intrinsic proposal bonus before blind or value-ranked actions, A1 hierarchical subgoal search only consumes the discovered first-contact trajectory, and A2 factored planner audits whether the novelty-selected actions have stable effects.",
      "method": "Episodic controllable-novelty policy family for L1 first contact",
      "residual_scope": "L1-first-contact wall: A1 reports wall_diagnosis=l1_first_contact and residual_cause_hypothesis=value_head_still_not_separating with generic first-win rate 0.04; A2 reports candidate_generation_coverage_factored=0.0 and residual_bridge_gap=experts_overfit_prefix, so .432 must change the action-proposal distribution before A1/A2 selection or planning.",
      "roadmap_candidate": "flagged_for_v432: controllable_novelty_e3_proposal_policy (arXiv:2002.06038 + arXiv:1810.12894 + arXiv:2603.02045)",
      "source_ids": [
        "2002.06038",
        "1810.12894",
        "2603.02045"
      ],
      "track": "ngu_rnd_controllable_novelty_e3_proposer"
    },
    {
      "fails_when": "the ensemble is undertrained on only a few public-game transitions, disagreement is high because of visual noise rather than mechanics, or empowerment rewards controllability that is unrelated to the L1 win.",
      "implement_cost_over_current_stack": "high: maintain a tiny ensemble over transition/effect predictions from live E3 traces, score short action sequences by predicted disagreement and causal controllability, then replay only the top frontier-expanding sequences through the existing harness.",
      "maps_to_current_stack": "live E3 explorer samples short sequences that are expected to expose new controllable effects, A1 hierarchical subgoal search is delayed until those effects create a reachable L1 contact, and A2 factored planner receives better transition evidence instead of overfit prefixes.",
      "method": "Plan2Explore-style disagreement frontier sampler with empowerment guard",
      "residual_scope": "L1-first-contact wall: A1 reports wall_diagnosis=l1_first_contact and residual_cause_hypothesis=value_head_still_not_separating with generic first-win rate 0.04; A2 reports candidate_generation_coverage_factored=0.0 and residual_bridge_gap=experts_overfit_prefix, so .432 must change the action-proposal distribution before A1/A2 selection or planning.",
      "roadmap_candidate": "flagged_for_v432: controllable_novelty_e3_proposal_policy (arXiv:2002.06038 + arXiv:1810.12894 + arXiv:2603.02045)",
      "source_ids": [
        "2005.05960",
        "2502.10077"
      ],
      "track": "model_disagreement_empowerment_frontier_sampler"
    },
    {
      "fails_when": "the behavior descriptors ignore the hidden winning mechanic, archive mutation destroys replayability, or the method rediscovers diverse near-misses without inserting the rare winning L1 prefix.",
      "implement_cost_over_current_stack": "medium: keep a MAP-Elites-style archive of replayable action prefixes using descriptors such as changed-cell topology, object motion class, HUD/register deltas, and novelty score; mutate prefixes only through actions that remain valid under live E3 replay.",
      "maps_to_current_stack": "live E3 explorer gets a diversified prefix generator instead of one depth-first action stream, A1 hierarchical subgoal search uses archive elites as first-contact candidates, and A2 factored planner checks whether elite descriptors correspond to reusable action effects.",
      "method": "Novelty/QD population over replayable action prefixes",
      "residual_scope": "L1-first-contact wall: A1 reports wall_diagnosis=l1_first_contact and residual_cause_hypothesis=value_head_still_not_separating with generic first-win rate 0.04; A2 reports candidate_generation_coverage_factored=0.0 and residual_bridge_gap=experts_overfit_prefix, so .432 must change the action-proposal distribution before A1/A2 selection or planning.",
      "roadmap_candidate": "flagged_for_v432: controllable_novelty_e3_proposal_policy (arXiv:2002.06038 + arXiv:1810.12894 + arXiv:2603.02045)",
      "source_ids": [
        "1712.06560",
        "1810.12894"
      ],
      "track": "novelty_qd_action_prefix_archive"
    },
    {
      "fails_when": "strategy text becomes another ungrounded subgoal layer, outcome reflection rewards plausible explanations rather than replayed state change, or mixed-temperature sampling spends the budget on duplicate mechanics.",
      "implement_cost_over_current_stack": "medium: generate a small batch of natural-language strategies at mixed temperatures, condition the action proposer on each strategy, and reflect only on replayed outcomes so the strategy pool is grounded in observed live E3 transitions.",
      "maps_to_current_stack": "live E3 explorer explores strategy-conditioned action streams, A1 hierarchical subgoal search is reused only after a strategy finds L1 contact, and A2 factored planner labels which strategies produced trustworthy action effects.",
      "method": "Strategy-guided exploration for language-action proposal diversity",
      "residual_scope": "L1-first-contact wall: A1 reports wall_diagnosis=l1_first_contact and residual_cause_hypothesis=value_head_still_not_separating with generic first-win rate 0.04; A2 reports candidate_generation_coverage_factored=0.0 and residual_bridge_gap=experts_overfit_prefix, so .432 must change the action-proposal distribution before A1/A2 selection or planning.",
      "roadmap_candidate": "flagged_for_v432: controllable_novelty_e3_proposal_policy (arXiv:2002.06038 + arXiv:1810.12894 + arXiv:2603.02045)",
      "source_ids": [
        "2603.02045",
        "2002.06038"
      ],
      "track": "strategy_guided_language_action_exploration"
    },
    {
      "fails_when": "the program overfits the first few prefixes, held-out transition trust is too sparse to reject brittle rules, hidden game state determines the effect, or the induced program explains effects but still cannot target the winning action.",
      "implement_cost_over_current_stack": "medium-high: synthesize small per-game action->effect programs from observed prefixes, reject programs that fail held-out transitions, and use surviving programs to propose mechanically relevant clicks or key actions rather than blind spatial sweeps.",
      "maps_to_current_stack": "live E3 explorer filters primitive proposals through induced action effects, A1 hierarchical subgoal search receives mechanically reachable first-contact prefixes, and A2 factored planner is narrowed to trusted program factors instead of composing prefix-overfit experts.",
      "method": "Program-synthesis action-effect induction for proposal pruning",
      "residual_scope": "L1-first-contact wall: A1 reports wall_diagnosis=l1_first_contact and residual_cause_hypothesis=value_head_still_not_separating with generic first-win rate 0.04; A2 reports candidate_generation_coverage_factored=0.0 and residual_bridge_gap=experts_overfit_prefix, so .432 must change the action-proposal distribution before A1/A2 selection or planning.",
      "roadmap_candidate": "flagged_for_v432: program_synthesis_action_effect_proposal_filter (arXiv:2505.10819 + arXiv:2102.11137)",
      "source_ids": [
        "2505.10819",
        "2102.11137"
      ],
      "track": "program_synthesis_action_effect_proposal_filter"
    }
  ],
  "note_path": "docs/research-notes/directed-exploration-sota-ingestion-2026-06-24.md",
  "preconditions_checked": {
    "agents_md_read": true,
    "arxiv_http_200_verified_ids": [
      "https://arxiv.org/abs/1712.06560",
      "https://arxiv.org/abs/1810.12894",
      "https://arxiv.org/abs/2002.06038",
      "https://arxiv.org/abs/2005.05960",
      "https://arxiv.org/abs/2102.11137",
      "https://arxiv.org/abs/2502.10077",
      "https://arxiv.org/abs/2505.10819",
      "https://arxiv.org/abs/2603.02045"
    ],
    "codex_md_read": true,
    "deep_research_invoked": false,
    "exp4673_artifact_read": true,
    "exp4673_note_read": true,
    "exp4676_artifact_read": true,
    "exp4677_artifact_read": true,
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
      "directed exploration intrinsic motivation novelty search empowerment interactive agents",
      "curiosity-driven exploration random network distillation episodic novelty reinforcement learning",
      "program synthesis action model induction interactive agents world models actions effects",
      "action effect prediction affordance induction interactive reinforcement learning program synthesis"
    ],
    "sweep_semscholar_rate_limited_queries": [
      "directed exploration intrinsic motivation novelty search empowerment interactive agents",
      "curiosity-driven exploration random network distillation episodic novelty reinforcement learning",
      "program synthesis action model induction interactive agents world models actions effects",
      "action effect prediction affordance induction interactive reinforcement learning program synthesis"
    ],
    "sweep_semscholar_used": true,
    "training_launched": false,
    "websearch_webfetch_top_sources": [
      "https://arxiv.org/abs/2002.06038",
      "https://arxiv.org/abs/1810.12894",
      "https://arxiv.org/abs/2005.05960",
      "https://arxiv.org/abs/1712.06560",
      "https://arxiv.org/abs/2502.10077",
      "https://arxiv.org/abs/2603.02045",
      "https://arxiv.org/abs/2102.11137",
      "https://arxiv.org/abs/2505.10819"
    ]
  },
  "random_seed": 4685
}
```

## Fresh-pass provenance

Read `AGENTS.md`, `CODEX.md`,
`results/experiment_4673_sota_ingestion_structural_deepening.json`,
`docs/research-notes/structural-deepening-sota-ingestion-2026-06-24.md`,
`results/experiment_4676_hierarchical_subgoal_search_live.json`,
`results/experiment_4677_poe_world_factored_subgoal_planner.json`,
`research-studying.md`, and `research-references.md`. The current stack is the
live E3 explorer with A1 hierarchical subgoal search and A2 factored planner
available only after a candidate trajectory exists. A1 closed with
`wall_diagnosis=l1_first_contact`, `value_head_still_not_separating`, and
generic first-win rate 0.04. A2 closed with
`candidate_generation_coverage_factored=0.0` and `experts_overfit_prefix`.
The `.432` scope is therefore not another selector; it is directed proposal
coverage so a winning L1 trajectory enters the pool.

Reliable-channel pass, not `/deep-research`:
- `curl -sf -o /dev/null https://huggingface.co/api/models`
- `.venv/bin/python scripts/sweep_clusters.py --help`
- `.venv/bin/python scripts/sweep_clusters.py 5 --max-results 8`
- `.venv/bin/python scripts/sweep_clusters.py 6 --max-results 8`
- `.venv/bin/python scripts/sweep_semscholar.py - --limit 8` with four focused queries
- low-concurrency WebSearch/WebFetch of the top directed-exploration and program-synthesis action-model papers
- direct arXiv HTTP checks for all cited IDs

Semantic Scholar returned HTTP 429 for the focused queries and no S2-only
source was promoted. Direct arXiv HTTP checks returned 200 for
arXiv:2002.06038, arXiv:1810.12894, arXiv:2005.05960, arXiv:1712.06560,
arXiv:2502.10077, arXiv:2603.02045, arXiv:2102.11137, and arXiv:2505.10819.
No live LLM inference, No training, No leaderboard submission, no model load,
and no live solve claim were run or made. `scripts/research_conductor.py`,
`ops/changelog.md`, and `ops/status.md` were not edited by this workflow.

## SOTA -> .432 directed-exploration mapping

## Episodic controllable-novelty policy family for L1 first contact

**Sources:** Never Give Up, arXiv:2002.06038; Exploration by Random Network
Distillation, arXiv:1810.12894; Strategy-Guided Exploration, arXiv:2603.02045.

**Mapping to current stack:** the live E3 explorer should score proposed
actions by controllable novelty before A1 or A2 ever sees a trajectory. A1
hierarchical subgoal search consumes the discovered first-contact trace only
after the proposal policy finds it. A2 factored planner audits whether the
novelty-selected actions express stable effects.

**Implementation cost over current stack:** medium. Add an embedding over
visible deltas/action effects, an episodic kNN novelty table, an RND-style
lifelong novelty score, and a small family of exploration temperatures under
the same replay gates.

**Fails when:** the embedding rewards cosmetic changes, KNN aliases mechanics,
or language strategy diversity produces plans that do not ground to valid ARC
actions.

## Plan2Explore-style disagreement frontier sampler with empowerment guard

**Sources:** Plan2Explore, arXiv:2005.05960; empowerment through causal
learning, arXiv:2502.10077.

**Mapping to current stack:** the live E3 explorer samples short sequences with
high predicted future novelty and controllability. A1 hierarchical subgoal
search is delayed until those sequences reveal an L1-contact candidate. A2
factored planner receives better transition evidence instead of composing
prefix-overfit experts.

**Implementation cost over current stack:** high. Maintain a small transition
ensemble, score short action sequences by predicted disagreement and causal
control, and replay only the top frontier-expanding sequences.

**Fails when:** transition data is too sparse, ensemble disagreement tracks
visual noise, or empowerment finds controllable states unrelated to the win.

## Novelty/QD population over replayable action prefixes

**Sources:** novelty-seeking ES/QD, arXiv:1712.06560; RND, arXiv:1810.12894.

**Mapping to current stack:** the live E3 explorer gets a replayable prefix
archive instead of a single depth-first stream. A1 hierarchical subgoal search
uses archive elites as first-contact candidates. A2 factored planner checks
whether elite descriptors correspond to reusable action effects.

**Implementation cost over current stack:** medium. Keep behavior descriptors
for changed-cell topology, object motion, HUD/register deltas, and novelty;
mutate only prefixes that survive the replay gate.

**Fails when:** descriptors miss the hidden mechanic, mutation breaks
replayability, or the archive diversifies near-misses without inserting the
rare winning L1 prefix.

## Strategy-guided exploration for language-action proposal diversity

**Sources:** Strategy-Guided Exploration, arXiv:2603.02045; Never Give Up,
arXiv:2002.06038.

**Mapping to current stack:** the live E3 explorer runs a small batch of
strategy-conditioned action streams. A1 hierarchical subgoal search starts only
after one strategy discovers L1 contact. A2 factored planner labels which
strategies produced trustworthy effects.

**Implementation cost over current stack:** medium. Generate concise strategy
sketches at mixed temperatures, condition action proposal on each strategy, and
reflect only on replayed outcomes.

**Fails when:** strategy text becomes another ungrounded subgoal layer, outcome
reflection rewards plausible explanation instead of state change, or the batch
duplicates one mechanic.

## Program-synthesis action-effect induction for proposal pruning

**Sources:** PoE-World, arXiv:2505.10819; model predictive program synthesis,
arXiv:2102.11137.

**Mapping to current stack:** the live E3 explorer filters primitive proposals
through per-game action-effect programs. A1 hierarchical subgoal search receives
mechanically reachable first-contact prefixes. A2 factored planner narrows to
trusted program factors instead of repeating the `experts_overfit_prefix`
failure.

**Implementation cost over current stack:** medium-high. Synthesize small
action->effect programs, reject programs that fail held-out transitions, and
use surviving programs to propose relevant clicks or key actions rather than
blind sweeps.

**Fails when:** the program overfits early prefixes, held-out transition trust
is too sparse, hidden state determines the effect, or the induced program
explains effects without targeting the winning action.

## Bottom line for the .432 roadmap

1. Build `flagged_for_v432: controllable_novelty_e3_proposal_policy` first.
   It directly attacks the `l1_first_contact` distribution gap: the current
   explorer reaches L1 on only 1/25 games, so the proposal distribution must
   be widened toward controllable novelty before selection helps.
2. Keep `flagged_for_v432: program_synthesis_action_effect_proposal_filter` as
   the second arm. It is the program-synthesis answer to blind clicks, but it
   must include held-out transition rejection because A2 already exposed the
   `experts_overfit_prefix` failure mode.
3. Treat Plan2Explore/empowerment and novelty/QD archives as support arms when
   the lightweight transition evidence is sufficient; both are valuable, but
   they can chase controllable non-wins if promoted without replay gates.

