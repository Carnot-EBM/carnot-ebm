# Intrinsic motivation / action-effect literature ingestion 2026-06-23

```json
{
  "citations_verified": {
    "1705.05363": {
      "http_status": 200,
      "title": "Curiosity-driven Exploration by Self-supervised Prediction",
      "url": "https://arxiv.org/abs/1705.05363"
    },
    "1810.12894": {
      "http_status": 200,
      "title": "Exploration by Random Network Distillation",
      "url": "https://arxiv.org/abs/1810.12894"
    },
    "2102.04399": {
      "http_status": 200,
      "title": "How to Stay Curious while Avoiding Noisy TVs using Aleatoric Uncertainty Estimation",
      "url": "https://arxiv.org/abs/2102.04399"
    },
    "2509.25438": {
      "http_status": 200,
      "title": "Beyond Noisy-TVs: Noise-Robust Exploration Via Learning Progress Monitoring",
      "url": "https://arxiv.org/abs/2509.25438"
    },
    "2512.24156": {
      "http_status": 200,
      "title": "Graph-Based Exploration for ARC-AGI-3 Interactive Reasoning Tasks",
      "url": "https://arxiv.org/abs/2512.24156"
    },
    "2601.10904": {
      "http_status": 200,
      "title": "ARC Prize 2025: Technical Report",
      "url": "https://arxiv.org/abs/2601.10904"
    },
    "2603.24621": {
      "http_status": 200,
      "title": "ARC-AGI-3: A New Challenge for Frontier Agentic Intelligence",
      "url": "https://arxiv.org/abs/2603.24621"
    },
    "2604.18701": {
      "http_status": 200,
      "title": "Curiosity-Critic: Cumulative Prediction Error Improvement as a Tractable Intrinsic Reward for World Model Training",
      "url": "https://arxiv.org/abs/2604.18701"
    },
    "2605.05138": {
      "http_status": 200,
      "title": "Executable World Models for ARC-AGI-3 in the Era of Coding Agents",
      "url": "https://arxiv.org/abs/2605.05138"
    }
  },
  "deep_research_not_used": true,
  "field_principles": {
    "deep_research_not_used": {
      "principle": "MUST be true -- /deep-research is banned in the autonomous loop; used sweep helpers + low-concurrency WebSearch/WebFetch."
    },
    "flagged_for_next_roadmap": {
      "principle": "the strongest method(s) flagged as candidate .428 inputs -- closes discover->ingest->plan->experiment."
    },
    "honest_verdict": {
      "principle": "terminal prefix; success: sota_ingestion_intrinsic_motivation_action_effect_mapped."
    },
    "inference_substrate": {
      "principle": "aggregation_from_upstream_artifacts -- literature read + synthesis, no model load (100us floor)."
    },
    "methods_mapped": {
      "principle": "the strongest 3-5 SOTA methods with REAL arXiv IDs + per-method implement-cost-over-current-stack + fails_when -- the actionable ingestion (no citation = fabrication)."
    },
    "note_path": {
      "principle": "docs/research-notes/intrinsic-motivation-action-effect-literature-2026-06-23.md -- the per-track note (the SOTA-Ingestion Cycle deliverable)."
    },
    "preconditions_checked": {
      "principle": "records network reachability verified; pre-empts fabricated citations."
    }
  },
  "flagged_for_next_roadmap": [
    "flagged_for_v428: curiosity_critic_learning_progress_dense_reward (arXiv:2604.18701 + arXiv:2509.25438)",
    "flagged_for_v428: noisy_tv_aware_action_effect_uncertainty_gate (arXiv:2102.04399 + arXiv:2509.25438)",
    "flagged_for_v428: clickability_action_effect_expansion_prior (arXiv:2601.10904 + arXiv:2603.24621)",
    "flagged_for_v428: graph_executable_world_model_action_effect_planner (arXiv:2512.24156 + arXiv:2605.05138)"
  ],
  "honest_verdict": "success: sota_ingestion_intrinsic_motivation_action_effect_mapped",
  "inference_substrate": "aggregation_from_upstream_artifacts",
  "methods_mapped": [
    {
      "fails_when": "the baseline critic mistakes hidden deterministic state for irreducible noise, logged transition classes are too sparse to learn the floor, or the intrinsic residual overwhelms first-win/action efficiency gates.",
      "implement_cost_over_current_stack": "medium-high: log the A2 action-effect model's per-transition prediction error, train a small A1 dense-curiosity critic to estimate the asymptotic/noise-floor error for each transition class, and feed only the positive learning-progress residual into expansion priority.",
      "maps_to_current_stack": "A1 dense-curiosity becomes a learnability estimator rather than a raw surprise score; A2 action-effect keeps the existing frame-change predictor but receives a dense reward for transitions whose error is still reducible.",
      "method": "Curiosity-Critic / LPM dense learning-progress reward",
      "roadmap_candidate": "flagged_for_v428: curiosity_critic_learning_progress_dense_reward (arXiv:2604.18701 + arXiv:2509.25438)",
      "source_ids": [
        "2604.18701",
        "2509.25438"
      ],
      "track": "dense_online_intrinsic_reward"
    },
    {
      "fails_when": "the variance head is undertrained, rare but decisive transitions look aleatoric early, or the guard suppresses the only probe that reveals a hidden register.",
      "implement_cost_over_current_stack": "medium: add mean/variance or previous-error heads beside the A2 action-effect predictor, down-weight high-variance transitions, and use the guarded score as an A1 dense-curiosity eligibility mask.",
      "maps_to_current_stack": "A1 dense-curiosity stops rewarding noisy-TV-like screen changes; A2 action-effect can still predict clickability, but only transitions classified as learnable receive exploration priority.",
      "method": "Aleatoric-noise guard for curiosity and action-effect rewards",
      "roadmap_candidate": "flagged_for_v428: noisy_tv_aware_action_effect_uncertainty_gate (arXiv:2102.04399 + arXiv:2509.25438)",
      "source_ids": [
        "2102.04399",
        "2509.25438"
      ],
      "track": "epistemic_vs_aleatoric_filtering"
    },
    {
      "fails_when": "raw prediction error chases stochastic animation, RND novelty decays before the useful mechanic is discovered, or the baseline improves coverage without reducing actions-to-first-win.",
      "implement_cost_over_current_stack": "low-to-medium: reuse the current frame-delta/action-effect tensors to train an inverse-dynamics feature space for ICM and a fixed-target embedding for RND, then compare both against the existing A1 dense curiosity score under matched action budgets.",
      "maps_to_current_stack": "A1 dense-curiosity gets a cheap baseline that validates whether any learned intrinsic reward beats raw prediction error; A2 action-effect uses the same transitions so no new environment substrate is needed.",
      "method": "ICM/RND prediction-error curiosity as a cheap control floor",
      "roadmap_candidate": "candidate_for_v428_control: icm_rnd_prediction_error_floor",
      "source_ids": [
        "1705.05363",
        "1810.12894"
      ],
      "track": "prediction_error_baseline_floor"
    },
    {
      "fails_when": "the predictor only ranks a fixed candidate pool, useful actions are not generated in the first place, or clickability improves frame change while failing to improve level completion and action count.",
      "implement_cost_over_current_stack": "low-to-medium: keep the current A2 action-effect CNN as a candidate expansion prior, train it on cached human/self-play transition rows, and gate it by first-win action efficiency instead of treating it as a post-hoc reranker.",
      "maps_to_current_stack": "A1 dense-curiosity supplies the dense learnability signal that tells the explorer when to keep probing; A2 action-effect turns the signal into fewer no-op or non-changing clicks under ARC-AGI-3's efficiency metric.",
      "method": "Clickability / action-effect expansion prior under ARC efficiency scoring",
      "roadmap_candidate": "flagged_for_v428: clickability_action_effect_expansion_prior (arXiv:2601.10904 + arXiv:2603.24621)",
      "source_ids": [
        "2601.10904",
        "2603.24621"
      ],
      "track": "clickability_action_effect_expansion"
    },
    {
      "fails_when": "state hashing aliases hidden registers, executable models pass prefix observations but fail held-out transitions, or graph exploration broadens coverage while spending more actions than the current live explorer.",
      "implement_cost_over_current_stack": "medium-high: persist a graph of tested state-action pairs, route untested but learnable edges through A1 dense-curiosity, and promote only verified A2 action-effect transitions into executable planning or shortest-path reuse.",
      "maps_to_current_stack": "A1 dense-curiosity chooses which frontier edges are worth testing; A2 action-effect supplies the transition predictions and the graph/world model prevents repeated actions that cannot change the state.",
      "method": "Graph/executable-world-model action-effect planner",
      "roadmap_candidate": "flagged_for_v428: graph_executable_world_model_action_effect_planner (arXiv:2512.24156 + arXiv:2605.05138)",
      "source_ids": [
        "2512.24156",
        "2605.05138"
      ],
      "track": "state_graph_world_model_action_effect"
    }
  ],
  "note_path": "docs/research-notes/intrinsic-motivation-action-effect-literature-2026-06-23.md",
  "preconditions_checked": {
    "agents_md_read": true,
    "arxiv_http_200_verified_ids": [
      "https://arxiv.org/abs/2604.18701",
      "https://arxiv.org/abs/2509.25438",
      "https://arxiv.org/abs/2102.04399",
      "https://arxiv.org/abs/1705.05363",
      "https://arxiv.org/abs/1810.12894",
      "https://arxiv.org/abs/2601.10904",
      "https://arxiv.org/abs/2603.24621",
      "https://arxiv.org/abs/2512.24156",
      "https://arxiv.org/abs/2605.05138"
    ],
    "bridge_diagnosis_note_read": true,
    "codex_md_read": true,
    "deep_research_invoked": false,
    "exp4625_artifact_read": true,
    "leaderboard_submission": false,
    "live_llm_inference": false,
    "model_load": false,
    "network_hf_models_reachable": true,
    "offline_live_bridge_note_read": true,
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
      "Curiosity-Critic cumulative prediction error improvement intrinsic reward 2604.18701",
      "intrinsic curiosity module random network distillation prediction error curiosity exploration",
      "learning progress epistemic aleatoric uncertainty exploration reinforcement learning",
      "ARC-AGI-3 clickability action effect CNN ARC Prize 2025 2601.10904",
      "Graph-Based Exploration ARC-AGI-3 2512.24156 Executable World Models 2605.05138"
    ],
    "sweep_semscholar_rate_limited_queries": [
      "Curiosity-Critic cumulative prediction error improvement intrinsic reward 2604.18701",
      "intrinsic curiosity module random network distillation prediction error curiosity exploration",
      "learning progress epistemic aleatoric uncertainty exploration reinforcement learning",
      "ARC-AGI-3 clickability action effect CNN ARC Prize 2025 2601.10904",
      "Graph-Based Exploration ARC-AGI-3 2512.24156 Executable World Models 2605.05138"
    ],
    "sweep_semscholar_used": true,
    "training_launched": false,
    "websearch_webfetch_top_sources": [
      "https://arxiv.org/abs/2604.18701",
      "https://arxiv.org/abs/2509.25438",
      "https://arxiv.org/abs/2102.04399",
      "https://arxiv.org/abs/1705.05363",
      "https://arxiv.org/abs/1810.12894",
      "https://arxiv.org/abs/2601.10904",
      "https://arxiv.org/abs/2603.24621",
      "https://arxiv.org/abs/2512.24156",
      "https://arxiv.org/abs/2605.05138"
    ]
  },
  "random_seed": 4637
}
```

## Fresh-pass provenance

Read `AGENTS.md`, `CODEX.md`,
`results/experiment_4625_sota_ingestion_offline_live_bridge.json`,
`docs/research-notes/offline-live-bridge-literature-2026-06-23.md`,
`docs/research-notes/arc-representation-not-the-bottleneck-2026-06-23.md`,
`research-studying.md`, and `research-references.md`. The filtered track was
the .427 headline open problem: GENERATE better live exploration through dense
online intrinsic-motivation / learning-progress signals plus action-effect
prediction for action efficiency, feeding candidate methods forward to .428.

Reliable-channel pass, not `/deep-research`:
- `.venv/bin/python -c "import urllib.request; urllib.request.urlopen('https://huggingface.co/api/models', timeout=10); print('net_ok')"`
- `.venv/bin/python scripts/sweep_clusters.py 5 --max-results 8`
- `.venv/bin/python scripts/sweep_clusters.py 6 --max-results 8`
- `.venv/bin/python scripts/sweep_semscholar.py - --limit 8` with five focused queries
- low-concurrency WebSearch/WebFetch of the top arXiv papers
- direct arXiv HTTP checks for all cited IDs

Semantic Scholar returned HTTP 429 for the five focused queries, so no
Semantic-Scholar-only source was promoted. Direct arXiv HTTP checks returned
200 for arXiv:2604.18701, arXiv:2509.25438, arXiv:2102.04399,
arXiv:1705.05363, arXiv:1810.12894, arXiv:2601.10904, arXiv:2603.24621,
arXiv:2512.24156, and arXiv:2605.05138. No live LLM inference, No training,
No leaderboard submission, no model load, and no live solve claim were run or
made. `scripts/research_conductor.py`, `ops/changelog.md`, and `ops/status.md`
were not edited by this workflow.

## SOTA -> experiment mapping

## Curiosity-Critic plus Learning Progress Monitoring

**Sources:** Curiosity-Critic, arXiv:2604.18701; Learning Progress Monitoring,
arXiv:2509.25438.

**Mapping to A1 dense-curiosity / A2 action-effect:** the current stack already
has an action-effect predictor, but raw prediction error is a bad dense reward
because it keeps paying for stochastic or already-learned transitions.
Curiosity-Critic turns the reward into improvement over an estimated
asymptotic error baseline, while Learning Progress Monitoring rewards model
improvement rather than novelty. The .428 implementation should attach this
critic to the A2 transition-error stream and expose the residual as the A1
dense-curiosity score.

**Implementation cost over current stack:** medium-high. Needs transition-error
logging, a small scalar baseline/error critic, held-out transition-class checks,
and action-efficiency no-regression gates.

**Fails when:** hidden deterministic state is misclassified as irreducible
noise, the baseline critic is too data-starved, or the dense reward becomes a
goal in itself rather than a way to reduce actions-to-first-win.

## Aleatoric-noise guard for curiosity rewards

**Sources:** Aleatoric Mapping Agents, arXiv:2102.04399; Learning Progress
Monitoring, arXiv:2509.25438.

**Mapping to A1 dense-curiosity / A2 action-effect:** the noisy-TV failure mode
is directly relevant to ARC sprites and UI effects: frame changes can be real
but not controllably useful. Add a variance/previous-error head beside the A2
action-effect predictor, then let A1 dense-curiosity prioritize only transitions
whose uncertainty appears reducible.

**Implementation cost over current stack:** medium. Adds one or two heads to
the predictor and a calibration split for stochastic vs learnable transitions.

**Fails when:** the guard suppresses rare discovery actions, the variance head
learns visual noise rather than action-conditioned unpredictability, or the
gate improves coverage but not action efficiency.

## ICM and RND as prediction-error control floors

**Sources:** ICM / self-supervised prediction curiosity, arXiv:1705.05363; RND,
arXiv:1810.12894.

**Mapping to A1 dense-curiosity / A2 action-effect:** ICM and RND are not the
recommended endpoint, but they are the control floor .428 must beat. Both can
be implemented over the same A2 frame-delta/action-effect tensors, giving A1 a
cheap check that learning-progress rewards really outperform raw prediction
error or novelty under matched action budgets.

**Implementation cost over current stack:** low-to-medium. Reuse transition
features, train the small curiosity heads, and compare first-win/action counts
against the current dense-curiosity loop.

**Fails when:** prediction error locks onto stochastic animations, RND novelty
vanishes too early, or a coverage gain does not translate into fewer actions.

## Clickability / action-effect expansion prior

**Sources:** ARC Prize 2025 technical report, arXiv:2601.10904; ARC-AGI-3
benchmark report, arXiv:2603.24621. Supplemental operational context from the
existing corpus: StochasticGoose-style clickability/action-effect code is not
promoted as an arXiv source, so the claim carried forward here is the
experiment design: use action-effect prediction under ARC's efficiency metric,
not a leaderboard-reproduction claim.

**Mapping to A1 dense-curiosity / A2 action-effect:** the .422/.427 lesson is
that ranking an already-bad candidate pool is not enough. A1 dense-curiosity
should decide where the explorer still has learnable action effects, and A2
should use the action-effect predictor during expansion so no-op actions are
not generated as often.

**Implementation cost over current stack:** low-to-medium. The predictor exists;
the work is moving it from post-hoc ranker to candidate-expansion prior and
measuring first-win/action efficiency.

**Fails when:** the useful action is absent from the candidate generator,
clickability predicts frame change without goal relevance, or the predictor is
trained on seen games and fails on hidden mechanics.

## Graph/executable-world-model action-effect planner

**Sources:** Graph-Based Exploration for ARC-AGI-3, arXiv:2512.24156;
Executable World Models for ARC-AGI-3, arXiv:2605.05138.

**Mapping to A1 dense-curiosity / A2 action-effect:** Graph-Based Exploration
shows the action-efficiency value of recording tested state-action pairs and
prioritizing untested edges; Executable World Models shows the higher-cost
variant where verified transitions become a planning substrate. For .428, A1
dense-curiosity picks which untested edges are learnable, and A2 action-effect
decides which predicted transitions are worth adding to the graph/model.

**Implementation cost over current stack:** medium-high. Needs stable state
hashing, tested-action ledgers, held-out transition verification, and a strict
rule that graph/world-model planning cannot increase action count at equal
first-win rate.

**Fails when:** hidden registers alias in the graph, executable models overfit
prefix observations, or systematic exploration spends more actions than the
current live explorer.

## Bottom line for the .428 roadmap

1. Build `flagged_for_v428: curiosity_critic_learning_progress_dense_reward`
   first: Curiosity-Critic arXiv:2604.18701 plus LPM arXiv:2509.25438 gives the
   dense reward that should replace raw surprise in A1.
2. Pair it immediately with
   `flagged_for_v428: noisy_tv_aware_action_effect_uncertainty_gate`:
   arXiv:2102.04399 and arXiv:2509.25438 are the guard against rewarding
   irreducible or useless frame changes.
3. Keep ICM arXiv:1705.05363 and RND arXiv:1810.12894 as matched-budget
   baselines, not as the final method, because .428 needs evidence that
   learning progress beats cheap raw prediction error.
4. Promote `flagged_for_v428: clickability_action_effect_expansion_prior` only
   if it changes candidate generation, not just ranking; the ARC reports
   arXiv:2601.10904 and arXiv:2603.24621 justify the action-efficiency target.
5. Use graph/executable planning, Graph-Based Exploration arXiv:2512.24156 plus
   Executable World Models arXiv:2605.05138, as the second-stage planner after
   dense curiosity and action-effect prediction pass no-regression gates.

