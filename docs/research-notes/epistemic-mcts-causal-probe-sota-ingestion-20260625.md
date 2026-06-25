# Epistemic-MCTS / causal-probe SOTA ingestion 20260625

```json
{
  "citations": {
    "2210.13455": {
      "http_status": 200,
      "title": "Epistemic Monte Carlo Tree Search",
      "url": "https://arxiv.org/abs/2210.13455"
    },
    "2309.08477": {
      "http_status": 200,
      "title": "Deep Multi-Agent Reinforcement Learning for Decentralized Active Hypothesis Testing",
      "url": "https://arxiv.org/abs/2309.08477"
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
    "2511.14262": {
      "http_status": 200,
      "title": "Object-Centric World Models for Causality-Aware Reinforcement Learning",
      "url": "https://arxiv.org/abs/2511.14262"
    },
    "2601.06604": {
      "http_status": 200,
      "title": "Object-Centric World Models Meet Monte Carlo Tree Search",
      "url": "https://arxiv.org/abs/2601.06604"
    },
    "2606.14418": {
      "http_status": 200,
      "title": "Causal Object-Centric Models for Planning with Monte Carlo Tree Search",
      "url": "https://arxiv.org/abs/2606.14418"
    }
  },
  "field_principles": {
    "citations": {
      "principle": "real arXiv IDs/URLs for every method claim -- an ingestion with no verifiable citations is fabrication per adversarial_verify discipline."
    },
    "flagged_for_next_roadmap": {
      "principle": "the strongest method(s) flagged_for_v436 -- closes the discover->ingest->plan loop so SOTA flows into the next milestone's experiments."
    },
    "honest_verdict": {
      "principle": "terminal prefix; success: sota_ingestion_epistemic_mcts_causal_probe_mapped."
    },
    "inference_substrate": {
      "principle": "aggregation_from_upstream_artifacts -- literature synthesis + WebFetch, no model load (100us floor)."
    },
    "methods_mapped": {
      "principle": "the 3-5 strongest methods, each with maps_to_current_stack + implement_cost_over_current_stack + fails_when -- the actionable ingestion (discover -> ingest -> plan)."
    },
    "note_path": {
      "principle": "docs/research-notes/epistemic-mcts-causal-probe-sota-ingestion-20260625.md -- the human-readable per-track synthesis."
    },
    "preconditions_checked": {
      "principle": "records the network-reachability check; pre-empts missing-resource fabrication."
    },
    "random_seed": {
      "principle": "determinism precondition (the search/synthesis seed)."
    },
    "reproducibility_checksum": {
      "principle": "content-addressed hash of the ingested source set."
    },
    "verifier_is_oracle": {
      "principle": "false -- a literature synthesis invokes no oracle."
    }
  },
  "flagged_for_next_roadmap": [
    "flagged_for_v436: epistemic_object_model_mcts_probe_planner (arXiv:2210.13455 + arXiv:2601.06604 + arXiv:2606.14418)",
    "flagged_for_v436: factored_interaction_causal_probe_bank (arXiv:2511.02225 + arXiv:2511.14262; guardrail arXiv:2511.06136)"
  ],
  "honest_verdict": "success: sota_ingestion_epistemic_mcts_causal_probe_mapped",
  "inference_substrate": "aggregation_from_upstream_artifacts",
  "methods_mapped": [
    {
      "fails_when": "epistemic uncertainty is miscalibrated, object state keys alias hidden registers, or rollout error compounds before live probes can correct the model.",
      "implement_cost_over_current_stack": "medium-high: add MCTS node statistics, rollout budgets, uncertainty propagation over candidate engines and ProductWorldModel factors, and a live-probe handoff compatible with the existing active-probe phase.",
      "maps_to_current_stack": "E3AgentPolicy calls an uncertainty-aware MCTS planner over arc_executable_world_model rollouts, replacing the single BFS plan_in_model call with nodes that can return either a solve action or an information-gain probe.",
      "method": "Epistemic object-model MCTS probe planner",
      "roadmap_candidate": "flagged_for_v436: epistemic_object_model_mcts_probe_planner (arXiv:2210.13455 + arXiv:2601.06604 + arXiv:2606.14418)",
      "source_ids": [
        "2210.13455",
        "2601.06604"
      ],
      "track": "epistemic_object_model_mcts_probe_planner"
    },
    {
      "fails_when": "ARC objects are not separable at logical-grid resolution, action targets cannot be grounded from click or keyboard data, or causal attention prioritizes visually salient but goal-irrelevant slots.",
      "implement_cost_over_current_stack": "medium-high: derive stable logical object slots from current frames, attach action-target metadata to candidate actions, and expose causal attention scores as a planner prior rather than a learned policy head.",
      "maps_to_current_stack": "E3AgentPolicy binds candidate ARC actions to object-like logical-grid slots before arc_executable_world_model rollouts, so MCTS can test which object interaction a click or movement action is meant to change.",
      "method": "Causal object-centric MCTS action-slot adapter",
      "roadmap_candidate": "support_for_v436: causal_object_mcts_action_slot_adapter (arXiv:2606.14418)",
      "source_ids": [
        "2606.14418"
      ],
      "track": "causal_object_mcts_action_slot_adapter"
    },
    {
      "fails_when": "object slots drift, causal labels alias hidden registers, interaction factors require longer interventions than the probe budget, or short prefixes make a spurious relation look causal.",
      "implement_cost_over_current_stack": "high: promote ProgrammaticExpert rows into a first-class causal factor bank, add confirm/refute ledgers, and let the factored planner compose only trusted interaction factors as subgoal transitions.",
      "maps_to_current_stack": "E3AgentPolicy proposes object-interaction hypotheses, arc_executable_world_model stores confirmed/refuted typed precondition/effect factors, and probe actions are selected to settle cause-effect relations before ProductWorldModel planning.",
      "method": "Factored interaction and causal probe bank",
      "roadmap_candidate": "flagged_for_v436: factored_interaction_causal_probe_bank (arXiv:2511.02225 + arXiv:2511.14262; guardrail arXiv:2511.06136)",
      "source_ids": [
        "2511.02225",
        "2511.14262"
      ],
      "track": "factored_interaction_causal_probe_bank"
    },
    {
      "fails_when": "the drift metric is too conservative and rejects all useful factors, or too permissive and lets unstable object rollouts pass into execution.",
      "implement_cost_over_current_stack": "low-medium: add off-path drift diagnostics, rejected-factor reasons, and plan invalidation when object-model predictions stay visually plausible but causally wrong.",
      "maps_to_current_stack": "E3AgentPolicy invalidates arc_executable_world_model plans when off-path object factors or causal relations drift under multi-object interactions, then routes the failure into the probe/factor ledger instead of executing a brittle plan.",
      "method": "Object-world-model drift and policy-breakage falsifier",
      "roadmap_candidate": "guardrail_for_v436: object_world_model_policy_breakage_falsifier (arXiv:2511.06136)",
      "source_ids": [
        "2511.06136"
      ],
      "track": "object_world_model_drift_policy_breakage_falsifier"
    }
  ],
  "note_path": "docs/research-notes/epistemic-mcts-causal-probe-sota-ingestion-20260625.md",
  "preconditions_checked": {
    "agents_md_read": true,
    "arc_competition_agent_read": true,
    "arc_executable_world_model_read": true,
    "arxiv_http_200_verified_ids": [
      "https://arxiv.org/abs/2210.13455",
      "https://arxiv.org/abs/2309.08477",
      "https://arxiv.org/abs/2506.01876",
      "https://arxiv.org/abs/2511.02225",
      "https://arxiv.org/abs/2511.06136",
      "https://arxiv.org/abs/2511.14262",
      "https://arxiv.org/abs/2601.06604",
      "https://arxiv.org/abs/2606.14418"
    ],
    "arxiv_reachable": true,
    "codex_md_read": true,
    "deep_research_invoked": false,
    "exp4722_artifact_read": true,
    "hypothesis_posterior_duplicate_skipped": true,
    "leaderboard_submission": false,
    "live_llm_inference": false,
    "model_load": false,
    "ops_docs_modified": false,
    "research_conductor_modified": false,
    "research_references_read": true,
    "solve_claim_made": false,
    "training_launched": false,
    "websearch_webfetch_top_sources": [
      "https://arxiv.org/abs/2506.01876",
      "https://arxiv.org/abs/2309.08477",
      "https://arxiv.org/abs/2210.13455",
      "https://arxiv.org/abs/2601.06604",
      "https://arxiv.org/abs/2606.14418",
      "https://arxiv.org/abs/2511.02225",
      "https://arxiv.org/abs/2511.14262",
      "https://arxiv.org/abs/2511.06136"
    ],
    "websearch_webfetch_used": true
  },
  "random_seed": 4734,
  "reproducibility_checksum": "sha256:6fc896f8a58bf8135927ec7c067079ff486b96b46c2034cf48885ebac5c63830",
  "verifier_is_oracle": false
}
```

## Fresh-pass provenance

Read `AGENTS.md`, `CODEX.md`,
`results/experiment_4722_sota_ingestion_active_probe_world_model.json`,
`research-references.md`, `python/carnot/agentic/arc_competition_agent.py`,
and `python/carnot/agentic/arc_executable_world_model.py`. The .435 A2 build
already implemented the hypothesis-posterior active-probe controller, so the
hypothesis-posterior controller is not re-flagged here. This note maps the
remaining .436 frontier: uncertainty-aware MCTS over executable world-model
rollouts, causal object-action slot binding, and factored interaction probes.

Reliable-channel pass, not `/deep-research`:
- `curl -sf -o /dev/null https://arxiv.org`
- focused WebSearch/WebFetch of the top epistemic-MCTS and causal-probe papers
- direct arXiv URL checks for all cited IDs

Direct arXiv HTTP checks returned 200 for arXiv:2506.01876, arXiv:2309.08477,
arXiv:2210.13455, arXiv:2601.06604, arXiv:2606.14418, arXiv:2511.02225,
arXiv:2511.14262, and arXiv:2511.06136. The first two are carried only as the
already-built .435 active-hypothesis baseline, not as a new .436 roadmap flag.
No live LLM inference, no model load, no training, no leaderboard submission,
and no solve claim were run or made. `scripts/research_conductor.py`,
`ops/changelog.md`, and `ops/status.md` were not edited by this workflow.

## SOTA -> .436 epistemic-MCTS / causal-probe mapping

## Epistemic object-model MCTS probe planner

**Sources:** Epistemic MCTS, arXiv:2210.13455; ObjectZero, arXiv:2601.06604.

**Mapping to current stack:** `E3AgentPolicy` currently calls
`arc_executable_world_model.plan_in_model` as a bounded BFS over one selected
engine. The v436 planner would replace that single-shot search with MCTS nodes
that propagate epistemic uncertainty across candidate engines and object-factor
rollouts, then return either the best solve action or the highest information
gain probe.

**Implementation cost over current stack:** medium-high. Add MCTS state keys,
rollout statistics, uncertainty propagation, and a probe-vs-act policy while
keeping the existing active-probe transition observer.

**Fails when:** uncertainty is uncalibrated, the object abstraction does not
reduce the branch factor, or model error compounds across rollouts faster than
live probes can correct it.

## Causal object-centric MCTS action-slot adapter

**Source:** COMET, arXiv:2606.14418.

**Mapping to current stack:** bind candidate ARC actions to object-like logical
slots before `arc_executable_world_model` rollouts, so MCTS can reason about
which object a click, drag, or movement action is expected to affect.

**Implementation cost over current stack:** medium-high. Derive stable logical
object slots from current frames, attach action-target metadata to candidate
actions, and expose causal attention scores as planner priors without adding a
learned policy head to the submitted path.

**Fails when:** object slots drift, action targets cannot be grounded from ARC
action data, or causal attention follows salient but goal-irrelevant objects.

## Factored interaction and causal probe bank

**Sources:** FIOC-WM, arXiv:2511.02225; STICA, arXiv:2511.14262.

**Mapping to current stack:** `E3AgentPolicy` already has a factored-planner
hook and `arc_executable_world_model` already has `ProgrammaticExpert`,
`ProductWorldModel`, and trusted-factor summaries. The v436 bank would promote
those rows into typed precondition/effect factors whose causal status is
confirmed or refuted by live probes.

**Implementation cost over current stack:** high. Add a factor schema, a
confirm/refute ledger, causal relation scoring, and planner composition only
over trusted factors.

**Fails when:** object slots drift, hidden registers alias relation labels,
short prefixes make spurious interactions look causal, or the probe budget is
too small for the needed intervention.

## Object-world-model drift and policy-breakage falsifier

**Source:** When Object-Centric World Models Meet Policy Learning,
arXiv:2511.06136.

**Mapping to current stack:** `E3AgentPolicy` should invalidate plans from
`arc_executable_world_model` when off-path object factors or causal relations
drift under multi-object interactions, then route those failures back into the
probe and factor ledgers instead of executing a brittle plan.

**Implementation cost over current stack:** low-medium. Add held-out off-path
drift diagnostics, rejected-factor reasons, and plan invalidation for visually
plausible but causally wrong rollouts.

**Fails when:** the drift metric rejects every useful factor or allows unstable
object rollouts into execution.

## Bottom line for the .436 roadmap

The strongest next candidate is
flagged_for_v436: epistemic_object_model_mcts_probe_planner
(arXiv:2210.13455 + arXiv:2601.06604 + arXiv:2606.14418). It is the direct
upgrade from the .435 posterior splitter to a planner that can choose between
acting and probing inside world-model rollouts.

The second v436 candidate is
flagged_for_v436: factored_interaction_causal_probe_bank
(arXiv:2511.02225 + arXiv:2511.14262), guarded by the drift/breakage falsifier
from arXiv:2511.06136. The honest bound is explicit: object-centric world
models can look visually stable and still break policy learning under off-path
multi-object interactions, so this ingestion makes no solve claim.
