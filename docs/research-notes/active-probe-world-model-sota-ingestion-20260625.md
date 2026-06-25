# Active-probe world-model SOTA ingestion 20260625

```json
{
  "citations": {
    "2007.07853": {
      "http_status": 200,
      "title": "Active World Model Learning with Progress Curiosity",
      "url": "https://arxiv.org/abs/2007.07853"
    },
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
    }
  },
  "field_principles": {
    "citations": {
      "principle": "real arXiv IDs/URLs for every method claim -- an ingestion with no verifiable citations is fabrication per adversarial_verify discipline."
    },
    "flagged_for_next_roadmap": {
      "principle": "the strongest method(s) flagged_for_v435 -- closes the discover->ingest->plan loop so SOTA flows into the next milestone's experiments."
    },
    "honest_verdict": {
      "principle": "terminal prefix; success: sota_ingestion_active_probe_world_model_mapped."
    },
    "inference_substrate": {
      "principle": "aggregation_from_upstream_artifacts -- literature synthesis + WebFetch, no model load (100us floor)."
    },
    "methods_mapped": {
      "principle": "the 3-5 strongest methods, each with maps_to_current_stack + implement_cost_over_current_stack + fails_when -- the actionable ingestion (discover -> ingest -> plan)."
    },
    "note_path": {
      "principle": "docs/research-notes/active-probe-world-model-sota-ingestion-20260625.md -- the human-readable per-track synthesis."
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
    "flagged_for_v435: hypothesis_posterior_active_probe_controller (arXiv:2506.01876 + arXiv:2309.08477)",
    "flagged_for_v435: epistemic_object_model_mcts_probe_planner (arXiv:2210.13455 + arXiv:2601.06604)",
    "flagged_for_v435: factored_interaction_causal_probe_bank (arXiv:2511.02225 + arXiv:2511.14262)"
  ],
  "honest_verdict": "success: sota_ingestion_active_probe_world_model_mapped",
  "inference_substrate": "aggregation_from_upstream_artifacts",
  "methods_mapped": [
    {
      "fails_when": "the true mechanic is outside the hypothesis class, the probe outcomes are aliased at logical-grid resolution, or the probe budget is spent identifying a rule that still does not imply the level goal.",
      "implement_cost_over_current_stack": "medium: add a hypothesis ledger, discriminating-probe scorer, posterior update from observed transitions, and a stop/act interface in the current induction phase without changing the environment API.",
      "maps_to_current_stack": "E3AgentPolicy keeps a small posterior over candidate goal and dynamics hypotheses, asks arc_executable_world_model to predict the transition each hypothesis expects, and chooses live actions that split that posterior before committing to a solve plan.",
      "method": "Hypothesis-posterior active probe controller",
      "roadmap_candidate": "flagged_for_v435: hypothesis_posterior_active_probe_controller (arXiv:2506.01876 + arXiv:2309.08477)",
      "source_ids": [
        "2506.01876",
        "2309.08477"
      ],
      "track": "hypothesis_posterior_active_probe_controller"
    },
    {
      "fails_when": "uncertainty is uncalibrated, object abstraction does not reduce branching factor, or model error compounds over rollouts faster than probes can correct.",
      "implement_cost_over_current_stack": "medium-high: replace the current bounded BFS-only planning path with MCTS nodes, rollout budgets, per-factor uncertainty, and a policy for when a high-uncertainty branch deserves a real live action.",
      "maps_to_current_stack": "E3AgentPolicy calls an uncertainty-aware MCTS planner over arc_executable_world_model rollouts, using object-level state keys when available and returning either a solve action or an information-gain probe.",
      "method": "Epistemic object-model MCTS probe planner",
      "roadmap_candidate": "flagged_for_v435: epistemic_object_model_mcts_probe_planner (arXiv:2210.13455 + arXiv:2601.06604)",
      "source_ids": [
        "2210.13455",
        "2601.06604"
      ],
      "track": "epistemic_object_model_mcts_probe_planner"
    },
    {
      "fails_when": "learning-progress reward chases dynamics that are easy to improve but irrelevant to the goal, or the signal degenerates into passive curiosity on visual noise.",
      "implement_cost_over_current_stack": "medium: record per-factor prediction error before and after each transition, add a progress estimate to frontier ordering, and cap it behind the existing target-level and budget controls.",
      "maps_to_current_stack": "E3AgentPolicy scores candidate probe actions by expected improvement in arc_executable_world_model factor prediction, preferring transitions that are learnable and mechanic-disambiguating rather than merely novel.",
      "method": "Progress-curiosity world-model improvement probes",
      "roadmap_candidate": "support_for_v435: progress_curiosity_probe_scheduler (arXiv:2007.07853)",
      "source_ids": [
        "2007.07853"
      ],
      "track": "progress_curiosity_world_model_probe_scheduler"
    },
    {
      "fails_when": "object slots drift, relation labels alias hidden registers, or short prefixes make a spurious interaction look causal.",
      "implement_cost_over_current_stack": "high: promote current programmatic experts into a first-class interaction factor schema, add causal relation scoring, and let the planner compose confirmed interactions as subgoals.",
      "maps_to_current_stack": "E3AgentPolicy proposes object-interaction hypotheses, arc_executable_world_model stores them as typed precondition/effect factors, and probe actions are selected to confirm or refute cause-effect relations.",
      "method": "Factored interaction and causal probe bank",
      "roadmap_candidate": "flagged_for_v435: factored_interaction_causal_probe_bank (arXiv:2511.02225 + arXiv:2511.14262)",
      "source_ids": [
        "2511.02225",
        "2511.14262"
      ],
      "track": "factored_interaction_causal_probe_bank"
    },
    {
      "fails_when": "the drift metric is too conservative and rejects every useful model, or too permissive and lets brittle object rollouts pass into execution.",
      "implement_cost_over_current_stack": "low-medium: add held-out off-path drift diagnostics, rejected-factor reasons, and plan invalidation when object-model predictions stay visually plausible but causally wrong.",
      "maps_to_current_stack": "E3AgentPolicy refuses plans from arc_executable_world_model when off-path object latents or relations drift under multi-object interactions, and routes those failures back into the active-probe ledger.",
      "method": "Object-world-model drift and policy-breakage falsifier",
      "roadmap_candidate": "guardrail_for_v435: object_world_model_policy_breakage_falsifier (arXiv:2511.06136)",
      "source_ids": [
        "2511.06136"
      ],
      "track": "object_world_model_drift_policy_breakage_falsifier"
    }
  ],
  "note_path": "docs/research-notes/active-probe-world-model-sota-ingestion-20260625.md",
  "preconditions_checked": {
    "agents_md_read": true,
    "arc_competition_agent_read": true,
    "arc_executable_world_model_read": true,
    "arxiv_http_200_verified_ids": [
      "https://arxiv.org/abs/2007.07853",
      "https://arxiv.org/abs/2210.13455",
      "https://arxiv.org/abs/2309.08477",
      "https://arxiv.org/abs/2506.01876",
      "https://arxiv.org/abs/2511.02225",
      "https://arxiv.org/abs/2511.06136",
      "https://arxiv.org/abs/2511.14262",
      "https://arxiv.org/abs/2601.06604"
    ],
    "arxiv_reachable": true,
    "codex_md_read": true,
    "deep_research_invoked": false,
    "exp4709_artifact_read": true,
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
      "https://arxiv.org/abs/2007.07853",
      "https://arxiv.org/abs/2511.02225",
      "https://arxiv.org/abs/2511.14262",
      "https://arxiv.org/abs/2601.06604",
      "https://arxiv.org/abs/2511.06136"
    ],
    "websearch_webfetch_used": true
  },
  "random_seed": 4722,
  "reproducibility_checksum": "sha256:04ed966b895bbec5ded3c9fa98266dbd369a413c1be173bbc3d5aad22ba0c96b",
  "verifier_is_oracle": false
}
```

## Fresh-pass provenance

Read `AGENTS.md`, `CODEX.md`,
`results/experiment_4709_sota_ingestion_structured_world_model.json`,
`research-references.md`, `python/carnot/agentic/arc_competition_agent.py`,
and `python/carnot/agentic/arc_executable_world_model.py`. The prior .433
ingestion already mapped the structured object-relational substrate for .434;
this note maps the next .435 frontier: active-probe / hypothesis-driven world-model induction,
where the agent acts to disambiguate goal and dynamics hypotheses before it
spends live actions on a solve path.

Reliable-channel pass, not `/deep-research`:
- `curl -sf -o /dev/null https://arxiv.org`
- focused WebSearch/WebFetch of the top active-probe and world-model papers
- direct arXiv URL checks for all cited IDs

Direct arXiv HTTP checks returned 200 for arXiv:2506.01876, arXiv:2309.08477,
arXiv:2210.13455, arXiv:2007.07853, arXiv:2511.02225, arXiv:2511.14262,
arXiv:2601.06604, and arXiv:2511.06136. No live LLM inference, no model load,
no training, no leaderboard submission, and no solve claim were run or made.
`scripts/research_conductor.py`, `ops/changelog.md`, and `ops/status.md` were
not edited by this workflow.

## SOTA -> .435 active-probe world-model mapping

## Hypothesis-posterior active probe controller

**Sources:** In-Context Pure Explorer, arXiv:2506.01876; decentralized active
hypothesis testing, arXiv:2309.08477.

**Mapping to current stack:** `E3AgentPolicy` keeps a small posterior over
candidate goal and dynamics hypotheses, asks `arc_executable_world_model` what
each hypothesis predicts for a candidate action, and picks probes that split the
posterior before committing to a solve plan.

**Implementation cost over current stack:** medium. Add a hypothesis ledger,
posterior updates from observed transitions, and a discriminating-probe scorer
inside the current induction/explore phase machine.

**Fails when:** the true mechanic is outside the hypothesis class, probes are
aliased at logical-grid resolution, or rule identification does not imply a
level-completion policy.

## Epistemic object-model MCTS probe planner

**Sources:** Epistemic MCTS, arXiv:2210.13455; ObjectZero, arXiv:2601.06604.

**Mapping to current stack:** `E3AgentPolicy` asks an uncertainty-aware MCTS
planner over `arc_executable_world_model` rollouts for either the next solve
action or the next information-gain probe.

**Implementation cost over current stack:** medium-high. The current planning
path is bounded BFS; this adds MCTS nodes, object-level state keys, rollout
budgets, uncertainty propagation, and a real-action policy for probe execution.

**Fails when:** uncertainty is uncalibrated, object abstractions do not shrink
the branch factor, or model error compounds faster than live probes can fix.

## Progress-curiosity world-model improvement probes

**Source:** Active World Model Learning with Progress Curiosity,
arXiv:2007.07853.

**Mapping to current stack:** `E3AgentPolicy` scores probes by expected
improvement in `arc_executable_world_model` factor prediction rather than by
passive novelty alone.

**Implementation cost over current stack:** medium. Record before/after
prediction error per factor and expose a bounded learning-progress term to
frontier ordering.

**Fails when:** progress reward chases learnable but goal-irrelevant dynamics,
or degenerates into curiosity over visual noise.

## Factored interaction and causal probe bank

**Sources:** FIOC-WM, arXiv:2511.02225; STICA, arXiv:2511.14262.

**Mapping to current stack:** `E3AgentPolicy` proposes object-interaction
hypotheses, `arc_executable_world_model` stores them as typed
precondition/effect factors, and probe actions confirm or refute the proposed
cause-effect relation.

**Implementation cost over current stack:** high. Promote programmatic experts
into a first-class interaction-factor schema, add causal relation scoring, and
let the planner compose confirmed interactions as subgoals.

**Fails when:** object slots drift, relation labels alias hidden registers, or
short prefixes make a spurious interaction look causal.

## Object-world-model drift and policy-breakage falsifier

**Source:** When Object-Centric World Models Meet Policy Learning,
arXiv:2511.06136.

**Mapping to current stack:** `E3AgentPolicy` refuses plans from
`arc_executable_world_model` when off-path object latents or relations drift
under multi-object interactions, and routes those failures back into the
active-probe ledger.

**Implementation cost over current stack:** low-medium. Add held-out off-path
drift diagnostics, rejected-factor reasons, and plan invalidation for visually
plausible but causally wrong rollouts.

**Fails when:** the drift metric rejects every useful induced model or permits
brittle object rollouts into execution.

## Bottom line for the .435 roadmap

The strongest .435 candidate is
flagged_for_v435: hypothesis_posterior_active_probe_controller
(arXiv:2506.01876 + arXiv:2309.08477). It converts the current passive
explore/induce cycle into active experiment selection: what action would most
disambiguate the goal or dynamics?

The planning companion is
flagged_for_v435: epistemic_object_model_mcts_probe_planner
(arXiv:2210.13455 + arXiv:2601.06604). The structural companion is
flagged_for_v435: factored_interaction_causal_probe_bank
(arXiv:2511.02225 + arXiv:2511.14262). The bound carried from
arXiv:2511.06136 is explicit: object-centric perception can still fail under
off-path policy interactions, so .435 must include the drift falsifier and make
no solve claim from literature alone.
