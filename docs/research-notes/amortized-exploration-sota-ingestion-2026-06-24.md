# Amortized-exploration SOTA ingestion 2026-06-24

```json
{
  "citations_verified": {
    "1802.07245": {
      "http_status": 200,
      "title": "Meta-Reinforcement Learning of Structured Exploration Strategies",
      "url": "https://arxiv.org/abs/1802.07245"
    },
    "1901.10995": {
      "http_status": 200,
      "title": "Go-Explore: a New Approach for Hard-Exploration Problems",
      "url": "https://arxiv.org/abs/1901.10995"
    },
    "2004.12919": {
      "http_status": 200,
      "title": "First return, then explore",
      "url": "https://arxiv.org/abs/2004.12919"
    },
    "2008.02790": {
      "http_status": 200,
      "title": "Decoupling Exploration and Exploitation for Meta-Reinforcement Learning without Sacrifices",
      "url": "https://arxiv.org/abs/2008.02790"
    },
    "2210.14215": {
      "http_status": 200,
      "title": "In-context Reinforcement Learning with Algorithm Distillation",
      "url": "https://arxiv.org/abs/2210.14215"
    },
    "2310.09971": {
      "http_status": 200,
      "title": "AMAGO: Scalable In-Context Reinforcement Learning for Adaptive Agents",
      "url": "https://arxiv.org/abs/2310.09971"
    },
    "2601.19810": {
      "http_status": 200,
      "title": "Unsupervised Learning of Efficient Exploration: Pre-training Adaptive Policies via Self-Imposed Goals",
      "url": "https://arxiv.org/abs/2601.19810"
    },
    "2603.03680": {
      "http_status": 200,
      "title": "MAGE: Meta-Reinforcement Learning for Language Agents toward Strategic Exploration and Exploitation",
      "url": "https://arxiv.org/abs/2603.03680"
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
      "principle": "the strongest method(s) flagged as candidate .433 inputs (flagged_for_v433) -- closes discover->ingest->plan->experiment."
    },
    "honest_verdict": {
      "principle": "terminal prefix; success: sota_ingestion_amortized_exploration_mapped."
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
    "flagged_for_v433: in_context_exploration_prior_from_first_contact_traces (arXiv:2210.14215 + arXiv:2310.09971 + arXiv:2601.19810)",
    "flagged_for_v433: arc_go_explore_return_then_explore_archive_upgrade (arXiv:1901.10995 + arXiv:2004.12919)"
  ],
  "honest_verdict": "success: sota_ingestion_amortized_exploration_mapped",
  "inference_substrate": "aggregation_from_upstream_artifacts",
  "methods_mapped": [
    {
      "fails_when": "the logged trajectories are too sparse, public-game successes encode game IDs rather than reusable mechanics, context windows omit decisive hidden state, or the distilled policy imitates late exploitation instead of first-contact probing.",
      "implement_cost_over_current_stack": "high: collect successful and near-miss first-contact trajectories across public games, serialize observations/actions/rewards/tool outcomes into long-context episodes, then train or fine-tune a small sequence policy that proposes the next exploration move before the per-game E3 proposer starts from an empty prior.",
      "maps_to_current_stack": "live E3 explorer receives a reusable cross-game action prior; A1 controllable-novelty proposal becomes a feature channel rather than the whole explorer; A2 program-synthesis action-effect filter labels which prior actions have trusted effects; arc_go_explore.py can replay prior-proposed prefixes from archive cells.",
      "method": "In-context exploration prior distilled from first-contact histories",
      "residual_scope": "Cross-game transfer wall: A1 controllable novelty ended with residual_cause_hypothesis=winning_prefix_still_not_proposed, reached_level=0, reproduced_levels=0, and chosen_submitted_config=unchanged; A2 program synthesis ended with coverage_delta=0.0, first_win_rate_delta=-0.04, and residual_bridge_gap=heldout_transitions_too_sparse. The deeper hidden-game transfer failure is that per-game directed exploration is re-derived from scratch, so even a per-public-game improvement can leave the scored hidden-game lane at 0.08.",
      "roadmap_candidate": "flagged_for_v433: in_context_exploration_prior_from_first_contact_traces (arXiv:2210.14215 + arXiv:2310.09971 + arXiv:2601.19810)",
      "source_ids": [
        "2210.14215",
        "2310.09971"
      ],
      "track": "in_context_rl_exploration_prior_from_first_contact_traces"
    },
    {
      "fails_when": "self-imposed goals reward visually rich but non-winning mechanics, structured noise remains too task-family-specific, or the curriculum never generates the rare stateful action combinations hidden games need.",
      "implement_cost_over_current_stack": "high: define ARC-compatible self-imposed goals from object motion, changed-cell topology, HUD/register deltas, and level-up proxies, then meta-train a policy with structured stochasticity so hidden games start with purposeful probing instead of flat action noise.",
      "maps_to_current_stack": "live E3 explorer samples from a learned exploration latent; A1 controllable-novelty proposal supplies controllable-effect embeddings as goals; A2 program-synthesis action-effect filter rejects brittle goal-action rules; arc_go_explore.py stores reached self-imposed-goal cells for return-and-extend.",
      "method": "Self-imposed-goal and structured-noise meta exploration prior",
      "residual_scope": "Cross-game transfer wall: A1 controllable novelty ended with residual_cause_hypothesis=winning_prefix_still_not_proposed, reached_level=0, reproduced_levels=0, and chosen_submitted_config=unchanged; A2 program synthesis ended with coverage_delta=0.0, first_win_rate_delta=-0.04, and residual_bridge_gap=heldout_transitions_too_sparse. The deeper hidden-game transfer failure is that per-game directed exploration is re-derived from scratch, so even a per-public-game improvement can leave the scored hidden-game lane at 0.08.",
      "roadmap_candidate": "flagged_for_v433: in_context_exploration_prior_from_first_contact_traces (arXiv:2210.14215 + arXiv:2310.09971 + arXiv:2601.19810)",
      "source_ids": [
        "2601.19810",
        "1802.07245"
      ],
      "track": "self_imposed_goal_meta_exploration_prior"
    },
    {
      "fails_when": "the exploration objective gathers information irrelevant to the executable win, language reflections hallucinate causal rules, or public-game multi-episode training overfits opponent/task identities rather than ARC mechanics.",
      "implement_cost_over_current_stack": "medium-high: split the current live loop into an explorer objective that gathers task-identifying transitions and an exploiter objective that attempts the level, then train/refit the language-agent reflection memory on multi-episode histories instead of one-game scratch plans.",
      "maps_to_current_stack": "live E3 explorer alternates explicit information-gathering and solve attempts; A1 controllable-novelty proposal is scored by whether it reveals task-relevant information; A2 program-synthesis action-effect filter provides exploitation facts; arc_go_explore.py supplies stable return states for repeated explore/exploit cycles.",
      "method": "Decoupled meta exploration/exploitation for language-agent adaptation",
      "residual_scope": "Cross-game transfer wall: A1 controllable novelty ended with residual_cause_hypothesis=winning_prefix_still_not_proposed, reached_level=0, reproduced_levels=0, and chosen_submitted_config=unchanged; A2 program synthesis ended with coverage_delta=0.0, first_win_rate_delta=-0.04, and residual_bridge_gap=heldout_transitions_too_sparse. The deeper hidden-game transfer failure is that per-game directed exploration is re-derived from scratch, so even a per-public-game improvement can leave the scored hidden-game lane at 0.08.",
      "roadmap_candidate": "not_primary_for_v433: useful only after a trajectory corpus exists",
      "source_ids": [
        "2008.02790",
        "2603.03680"
      ],
      "track": "decoupled_meta_explore_exploit_language_agent"
    },
    {
      "fails_when": "cell descriptors alias hidden registers, replay cannot restore the chosen state, the archive expands many dead cells without a goal gradient, or stochastic live conditions break deterministic offline returns.",
      "implement_cost_over_current_stack": "medium: harden the existing arc_go_explore.py archive with cross-game cell descriptors, state-restore/replay checks, under-visited-cell scheduling, and a bridge that feeds archive prefixes back into the live E3/A1/A2 proposal stack.",
      "maps_to_current_stack": "live E3 explorer gets replayable prefixes instead of restarting every hidden game from scratch; A1 controllable-novelty proposal scores post-return actions; A2 program-synthesis action-effect filter validates archive extensions; arc_go_explore.py is the existing return-then-explore implementation to upgrade.",
      "method": "Return-then-explore archive upgrade for reusable first-contact state coverage",
      "residual_scope": "Cross-game transfer wall: A1 controllable novelty ended with residual_cause_hypothesis=winning_prefix_still_not_proposed, reached_level=0, reproduced_levels=0, and chosen_submitted_config=unchanged; A2 program synthesis ended with coverage_delta=0.0, first_win_rate_delta=-0.04, and residual_bridge_gap=heldout_transitions_too_sparse. The deeper hidden-game transfer failure is that per-game directed exploration is re-derived from scratch, so even a per-public-game improvement can leave the scored hidden-game lane at 0.08.",
      "roadmap_candidate": "flagged_for_v433: arc_go_explore_return_then_explore_archive_upgrade (arXiv:1901.10995 + arXiv:2004.12919)",
      "source_ids": [
        "1901.10995",
        "2004.12919"
      ],
      "track": "go_explore_return_then_explore_archive"
    }
  ],
  "note_path": "docs/research-notes/amortized-exploration-sota-ingestion-2026-06-24.md",
  "preconditions_checked": {
    "agents_md_read": true,
    "arc_go_explore_read": true,
    "arxiv_http_200_verified_ids": [
      "https://arxiv.org/abs/1802.07245",
      "https://arxiv.org/abs/1901.10995",
      "https://arxiv.org/abs/2004.12919",
      "https://arxiv.org/abs/2008.02790",
      "https://arxiv.org/abs/2210.14215",
      "https://arxiv.org/abs/2310.09971",
      "https://arxiv.org/abs/2601.19810",
      "https://arxiv.org/abs/2603.03680"
    ],
    "codex_md_read": true,
    "deep_research_invoked": false,
    "exp4685_artifact_read": true,
    "exp4685_note_read": true,
    "exp4688_artifact_read": true,
    "exp4689_artifact_read": true,
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
      "amortized meta exploration learned exploration prior in-context reinforcement learning adaptive agents",
      "algorithm distillation in-context reinforcement learning exploration trajectories",
      "go-explore first return then explore return then explore archive reinforcement learning",
      "meta reinforcement learning exploration policy prior sparse reward"
    ],
    "sweep_semscholar_rate_limited_queries": [
      "amortized meta exploration learned exploration prior in-context reinforcement learning adaptive agents",
      "algorithm distillation in-context reinforcement learning exploration trajectories",
      "go-explore first return then explore return then explore archive reinforcement learning",
      "meta reinforcement learning exploration policy prior sparse reward"
    ],
    "sweep_semscholar_used": true,
    "training_launched": false,
    "websearch_webfetch_top_sources": [
      "https://arxiv.org/abs/2310.09971",
      "https://arxiv.org/abs/2210.14215",
      "https://arxiv.org/abs/2601.19810",
      "https://arxiv.org/abs/2008.02790",
      "https://arxiv.org/abs/1802.07245",
      "https://arxiv.org/abs/2004.12919",
      "https://arxiv.org/abs/1901.10995",
      "https://arxiv.org/abs/2603.03680"
    ]
  },
  "random_seed": 4697
}
```

## Fresh-pass provenance

Read `AGENTS.md`, `CODEX.md`,
`results/experiment_4685_sota_ingestion_directed_exploration.json`,
`docs/research-notes/directed-exploration-sota-ingestion-2026-06-24.md`,
`results/experiment_4688_controllable_novelty_proposal_policy_live.json`,
`results/experiment_4689_program_synthesis_action_effect_proposal_filter.json`,
`python/carnot/agentic/arc_go_explore.py`, `research-studying.md`, and
`research-references.md`. The .432 A1 arm closed with
`winning_prefix_still_not_proposed`, no reproduced level, and unchanged
submitted config. The .432 A2 arm closed with `heldout_transitions_too_sparse`,
coverage delta 0.0, and unchanged submitted config. The .433 scope is therefore
hidden-game transfer: first-contact behavior must be amortized across games instead
of being rediscovered from scratch on each scored hidden game.

Reliable-channel pass, not `/deep-research`:
- `curl -sf -o /dev/null https://huggingface.co/api/models`
- `.venv/bin/python scripts/sweep_clusters.py --help`
- `.venv/bin/python scripts/sweep_clusters.py 5 --max-results 8`
- `.venv/bin/python scripts/sweep_clusters.py 6 --max-results 8`
- `.venv/bin/python scripts/sweep_semscholar.py - --limit 8` with four focused queries
- low-concurrency WebSearch/WebFetch of the top amortized/meta exploration and Go-Explore papers
- direct arXiv HTTP checks for all cited IDs

Semantic Scholar returned HTTP 429 for the focused queries and no S2-only
source was promoted. Direct arXiv HTTP checks returned 200 for
arXiv:2210.14215, arXiv:2310.09971, arXiv:2601.19810, arXiv:1802.07245,
arXiv:2008.02790, arXiv:2603.03680, arXiv:1901.10995, and arXiv:2004.12919.
No live LLM inference, no training, no leaderboard submission, no model load,
and no live solve claim were run or made. `scripts/research_conductor.py`,
`ops/changelog.md`, and `ops/status.md` were not edited by this workflow.

## SOTA -> .433 amortized-exploration mapping

## In-context exploration prior distilled from first-contact histories

**Sources:** Algorithm Distillation, arXiv:2210.14215; AMAGO, arXiv:2310.09971.

**Mapping to current stack:** train a cross-game sequence policy over
observation/action/reward/tool histories so the live E3 explorer begins hidden
games with a reusable probing prior. A1 controllable-novelty proposal becomes a
feature channel. A2 program-synthesis action-effect filter supplies trusted
effect labels. `arc_go_explore.py` can replay the prior's proposed prefixes
from archive cells.

**Implementation cost over current stack:** high. The current stack logs enough
per-game traces for evaluation, but it does not yet curate a cross-game
trajectory dataset or train a sequence policy. The required build is a compact
trajectory schema plus an offline distillation/fine-tuning job.

**Fails when:** the first-contact corpus is too sparse, public-game identifiers
leak into the prior, or the model imitates late solve exploitation rather than
early exploration.

## Self-imposed-goal and structured-noise meta exploration prior

**Sources:** ULEE self-imposed goals, arXiv:2601.19810; MAESN structured
exploration, arXiv:1802.07245.

**Mapping to current stack:** generate cross-game self-imposed goals from
controllable cell changes, object motion, register/HUD deltas, and level-up
proxies. The live E3 explorer samples a learned exploration latent, A1 scores
controllable novelty against those goals, A2 rejects brittle rules, and
`arc_go_explore.py` stores reached goal cells for return-and-extend.

**Implementation cost over current stack:** high. It needs a pretraining
curriculum and a goal-descriptor vocabulary, but it directly addresses the
hidden-game transfer problem rather than retuning one game at a time.

**Fails when:** the goal vocabulary rewards visual churn, structured noise is
too family-specific, or the curriculum never reaches the stateful action
combinations that hidden games score.

## Decoupled meta exploration/exploitation for language-agent adaptation

**Sources:** DREAM, arXiv:2008.02790; MAGE, arXiv:2603.03680.

**Mapping to current stack:** split the live agent into an information-gathering
phase and a solve/exploitation phase. The live E3 explorer gathers
task-identifying transitions, A1 scores which probes reveal controllable
information, A2 converts trusted effects into exploitation facts, and
`arc_go_explore.py` provides stable return states for repeated cycles.

**Implementation cost over current stack:** medium-high. It reuses the existing
live loop but needs multi-episode memory/reflection training and a clean
separation between task-identification rewards and level-completion rewards.

**Fails when:** the exploration objective optimizes irrelevant information,
language reflections invent causal rules, or public-game multi-episode training
overfits task identities.

## Return-then-explore archive upgrade for reusable first-contact state coverage

**Sources:** Go-Explore, arXiv:1901.10995; First return, then explore,
arXiv:2004.12919.

**Mapping to current stack:** harden `arc_go_explore.py` so the archive is a
first-class producer of replayable prefixes. The live E3 explorer can return to
under-explored cells, A1 controllable-novelty proposal scores the post-return
actions, and A2 program-synthesis action-effect filter validates archive
extensions.

**Implementation cost over current stack:** medium. The scaffold already exists,
but it needs cross-game cell descriptors, restore/replay verification, and a
bridge that feeds archive prefixes into the submitted live stack.

**Fails when:** hidden registers alias into the same cell, replay cannot restore
the selected state, or the archive expands many dead cells without a goal
gradient.

## Bottom line for the .433 roadmap

The strongest .433 input is
flagged_for_v433: in_context_exploration_prior_from_first_contact_traces
(arXiv:2210.14215 + arXiv:2310.09971 + arXiv:2601.19810). It directly attacks
the cross-game transfer failure by amortizing successful and near-miss
first-contact trajectories into a reusable policy.

The structural companion is
flagged_for_v433: arc_go_explore_return_then_explore_archive_upgrade
(arXiv:1901.10995 + arXiv:2004.12919). It is cheaper because
`arc_go_explore.py` already exists, and it gives the exploration prior stable
return points rather than forcing every hidden-game attempt to restart from the
initial state.
