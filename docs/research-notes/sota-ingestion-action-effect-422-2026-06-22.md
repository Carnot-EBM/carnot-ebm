# Action-effect SOTA ingestion .422 - 2026-06-22

```json
{
  "citations_verified": {
    "2502.00271": {
      "http_status": 200,
      "title": "Scaling Flaws of Verifier-Guided Search in Mathematical Reasoning",
      "url": "https://arxiv.org/abs/2502.00271"
    },
    "2502.18407": {
      "http_status": 200,
      "title": "AgentRM: Enhancing Agent Generalization with Reward Modeling",
      "url": "https://arxiv.org/abs/2502.18407"
    },
    "2504.16828": {
      "http_status": 200,
      "title": "Process Reward Models That Think",
      "url": "https://arxiv.org/abs/2504.16828"
    },
    "2601.22607": {
      "http_status": 200,
      "title": "From Self-Evolving Synthetic Data to Verifiable-Reward RL: Post-Training Multi-turn Interactive Tool-Using Agents",
      "url": "https://arxiv.org/abs/2601.22607"
    },
    "2602.01070": {
      "http_status": 200,
      "title": "What If We Allocate Test-Time Compute Adaptively?",
      "url": "https://arxiv.org/abs/2602.01070"
    },
    "2603.24621": {
      "http_status": 200,
      "title": "ARC-AGI-3: A New Challenge for Frontier Agentic Intelligence",
      "url": "https://arxiv.org/abs/2603.24621"
    }
  },
  "field_principles": {
    "citations_verified": "every method claim cites a verifiable arXiv ID/URL -- the no-fabrication bar (same as any results artifact).",
    "field_principles": "principle annotations for every top-level artifact field.",
    "flagged_for_next_roadmap": "the strongest method flagged as a .423 candidate -- closes the discover->ingest->plan loop.",
    "honest_verdict": "terminal prefix; complete: sota_ingestion_action_effect_mapped.",
    "inference_substrate": "aggregation_from_upstream_artifacts -- literature synthesis, no model load (100us floor).",
    "methods_mapped": "the 3-5 strongest methods with REAL arXiv IDs -- the shoulders-of-giants anti-rederivation check.",
    "preconditions_checked": "records resources verified; pre-empts missing-resource fabrication.",
    "random_seed": "bare integer seed for reproducible artifact generation.",
    "research_note_path": "repo-relative markdown path for deterministic parsing."
  },
  "flagged_for_next_roadmap": "flagged_for_v423: use a StochasticGoose-style learned action-effect model as the candidate-expansion prior, then allocate Exp 4569 best-first frontier budget with adaptive PRM guidance and scaling-flaw controls (arXiv:2603.24621 + arXiv:2602.01070 + arXiv:2502.00271)",
  "honest_verdict": "complete: sota_ingestion_action_effect_mapped",
  "inference_substrate": "aggregation_from_upstream_artifacts",
  "methods_mapped": [
    {
      "fails_when": "the predictor is evaluated only on frame-change classification, allowed to suppress rare necessary actions without a positive control, or left as a final sorter over a pool where the winning action is absent.",
      "method": "StochasticGoose-style learned frame-change clickability predictor",
      "source_ids": [
        "2603.24621"
      ],
      "takes_over_current_a1_a2_mechanisms": "Exp 4568 trained and wired a pooled clickability/action-effect predictor but ended as an honest no-gain null because it only reranked existing candidates. This method keeps the ARC-AGI-3 action-efficiency target and changes the role: train a coarse CNN or equivalent frame-change model to decide which action families and click locations enter the candidate set before the explorer spends actions.",
      "target_track": "a1_action_effect_predictor",
      "v423_candidate": "flagged_for_v423: StochasticGoose-style action-effect model used as candidate expansion prior, not only as Exp 4568 reranker"
    },
    {
      "fails_when": "reward scores replace the exact reproduction gate, the reward model is trained on the held-out games used for transfer claims, or branch pruning is irreversible when the verifier is uncertain.",
      "method": "AgentRM generalizable reward model for agent search",
      "source_ids": [
        "2502.18407",
        "2502.00271"
      ],
      "takes_over_current_a1_a2_mechanisms": "Exp 4569 promoted a learned DiscriminativeVerifier into frontier expansion but still nulled on generic transfer. AgentRM takes over that Exp 4569 control point by scoring partial agent trajectories for test-time search and beam/frontier selection, while the scaling-flaws paper supplies the guardrail against verifier-only pruning on hard or out-of-distribution branches.",
      "target_track": "a2_verifier_guided_expansion",
      "v423_candidate": "flagged_for_v423: AgentRM-style trajectory reward for bounded Exp 4569 candidate expansion with anti-pruning guardrails"
    },
    {
      "fails_when": "long verification chains consume the first-contact action budget, local step plausibility is mistaken for final progress, or a single ThinkPRM score is allowed to eliminate every alternative branch.",
      "method": "ThinkPRM generative process verifier for expansion quality",
      "source_ids": [
        "2504.16828",
        "2502.00271"
      ],
      "takes_over_current_a1_a2_mechanisms": "Exp 4569 used a cheap discriminative score but lacked a stronger explanation-based process verifier for ambiguous branches. ThinkPRM takes over the expensive-check tier: ask for generative step verification only on high-upside candidate branches, then feed the result into best-first expansion while keeping the scaling-flaws caution as a random-priority and repeated-sampling control requirement.",
      "target_track": "a2_verifier_guided_expansion",
      "v423_candidate": "flagged_for_v423: ThinkPRM only as sparse expensive check inside Exp 4569 expansion, never as sole branch killer"
    },
    {
      "fails_when": "the controller spends extra compute without lowering actions to first level-up, compares against an easier baseline budget, or inherits verifier-guided search scaling flaws by pruning valid paths on weak PRM evidence.",
      "method": "adaptive PRM-guided best-first candidate expansion",
      "source_ids": [
        "2602.01070",
        "2502.00271"
      ],
      "takes_over_current_a1_a2_mechanisms": "Exp 4569 already tries verifier-guided expansion, but its null shows the score must allocate expansion budget more carefully. Adaptive test-time compute allocation takes over the live best-first scheduler: aggregate process rewards to choose which frontier nodes to expand, when to widen, and when to fall back to less verifier-dependent search.",
      "target_track": "a2_verifier_guided_expansion",
      "v423_candidate": "flagged_for_v423: adaptive PRM-guided expansion scheduler over Exp 4569 with repeated-sampling/random-priority controls"
    },
    {
      "fails_when": "synthetic traces leak game identity, generated checkers are not execution-tested, or the self-evolving loop optimizes the verifier instead of held-out action efficiency.",
      "method": "self-evolving verifiable-reward data for predictor and verifier refresh",
      "source_ids": [
        "2601.22607",
        "2502.00271"
      ],
      "takes_over_current_a1_a2_mechanisms": "Exp 4568 and Exp 4569 both depend on fixed local corpora. Self-evolving verifiable-reward data takes over the refresh loop: generate new interaction traces with executable per-instance checks, add hard negatives where verifier-guided search pruned valid paths, and retrain the action-effect prior and expansion verifier only after held-out checks stay separate.",
      "target_track": "a2_verifier_guided_expansion",
      "v423_candidate": "flagged_for_v423: self-evolving checked traces to refresh the Exp 4568 predictor and Exp 4569 expansion verifier"
    }
  ],
  "preconditions_checked": {
    "agents_md_read": true,
    "arxiv_api_reachable": true,
    "arxiv_http_200_verified_ids": [
      "2603.24621",
      "2502.18407",
      "2504.16828",
      "2502.00271",
      "2602.01070",
      "2601.22607"
    ],
    "codex_md_read": true,
    "deep_research_invoked": false,
    "exp4568_artifact_read": true,
    "exp4568_spec_read": true,
    "exp4569_artifact_read": true,
    "exp4569_spec_read": true,
    "leaderboard_submission": false,
    "live_llm_inference": false,
    "live_solve_claim": false,
    "ops_docs_modified": false,
    "research_conductor_modified": false,
    "research_references_filtered": true,
    "research_studying_filtered": true,
    "research_studying_updated": true,
    "sweep_clusters_help_exit_0": true,
    "sweep_clusters_urls": [
      "http://export.arxiv.org/api/query?search_query=(abs:\"affordance\"+OR+abs:\"action+effect\"+OR+abs:\"clickability\"+OR+abs:\"frame+prediction\"+OR+abs:\"intrinsic+motivation\"+OR+abs:\"directed+exploration\"+OR+abs:\"novelty+search\")+AND+(abs:\"reinforcement+learning\"+OR+abs:\"agent\"+OR+abs:\"exploration\"+OR+abs:\"interactive+environment\"+OR+abs:\"ARC\")&start=0&max_results=8&sortBy=submittedDate&sortOrder=descending",
      "http://export.arxiv.org/api/query?search_query=(abs:\"neural+guided+search\"+OR+abs:\"learned+heuristic\"+OR+abs:\"value+guided+search\"+OR+abs:\"program+induction\"+OR+abs:\"world+model\"+OR+abs:\"goal+induction\")+AND+(abs:\"planning\"+OR+abs:\"agent\"+OR+abs:\"reasoning\"+OR+abs:\"reinforcement+learning\")&start=0&max_results=8&sortBy=submittedDate&sortOrder=descending"
    ],
    "sweep_clusters_used": true,
    "sweep_semscholar_arxiv_ids": [],
    "sweep_semscholar_queries": [
      "ARC-AGI-3 clickability action effect frame change CNN interactive agent exploration",
      "verifier guided candidate expansion process reward model AgentRM ThinkPRM adaptive test time compute"
    ],
    "sweep_semscholar_rate_limited_queries": [
      "ARC-AGI-3 clickability action effect frame change CNN interactive agent exploration",
      "verifier guided candidate expansion process reward model AgentRM ThinkPRM adaptive test time compute"
    ],
    "sweep_semscholar_used": true,
    "training_launched": false,
    "websearch_webfetch_top_sources": [
      "https://arxiv.org/abs/2603.24621",
      "https://arxiv.org/abs/2502.18407",
      "https://arxiv.org/abs/2504.16828",
      "https://arxiv.org/abs/2502.00271",
      "https://arxiv.org/abs/2602.01070",
      "https://arxiv.org/abs/2601.22607",
      "https://github.com/DriesSmit/ARC3-solution"
    ]
  },
  "random_seed": 4577,
  "research_note_path": "docs/research-notes/sota-ingestion-action-effect-422-2026-06-22.md"
}
```

Reliable channel only: `scripts/sweep_clusters.py`, `scripts/sweep_semscholar.py`,
arXiv abs-page HTTP-200 checks, and low-concurrency WebSearch/WebFetch of seven
top action-effect / clickability / verifier-guided candidate expansion sources.
Preconditions passed before any claim was promoted:
`.venv/bin/python scripts/sweep_clusters.py --help` exited zero and
`curl -sf -o /dev/null https://export.arxiv.org/api/query?search_query=all:test`
confirmed arXiv API reachability. `scripts/sweep_clusters.py 5 --max-results 8`
and `scripts/sweep_clusters.py 6 --max-results 8` emitted the focused
action-effect and learned-search cluster URLs. `scripts/sweep_semscholar.py` ran
two focused queries and returned HTTP 429 on both, so no S2-only claim was
promoted. No `/deep-research` call was made. No training, live LLM inference,
leaderboard submission, or live solve was launched. No ops/status/traceability
files or `scripts/research_conductor.py` were modified.

Already-discovered corpus read through a learned action-effect / clickability /
exploration-efficiency and verifier-guided candidate expansion filter:
`research-studying.md`, `research-references.md`,
`openspec/capabilities/arc-human-replay-frame-change/spec.md` at
`REQ-ARC-FCP-4568`, `openspec/capabilities/capstone/spec.md` at
`REQ-CAPSTONE-4569`, `results/experiment_4568_clickability_action_effect_predictor.json`,
and `results/experiment_4569_verifier_guided_expansion.json`. Exp 4568 was an
honest clickability null: the local predictor learned a positive control but did
not reduce held-out actions because it reranked a fixed candidate pool. Exp 4569
was also a no-value/null or negative transfer for verifier-guided candidate
expansion: candidate generation stayed the residual bottleneck, the
random-priority control did not pass, and the winner was still absent from most
frontiers.

Sources checked: arXiv:2603.24621, arXiv:2502.18407, arXiv:2504.16828, arXiv:2502.00271, arXiv:2602.01070, arXiv:2601.22607. The WebSearch/WebFetch pass also checked the
StochasticGoose/Tufa implementation source at https://github.com/DriesSmit/ARC3-solution.

## Per-Method Mapping

- **StochasticGoose-style learned frame-change clickability predictor**
  (arXiv:2603.24621): the strongest A1 lesson is not to keep Exp 4568 as a
  final sorter. The action-effect model must decide which action families and
  click locations enter the search frontier before first-contact actions are
  spent. It fails when measured only as frame-change accuracy or when it prunes
  rare necessary actions without a positive-control and recall guard.
- **AgentRM generalizable reward model for agent search** (arXiv:2502.18407,
  arXiv:2502.00271): take over Exp 4569's expansion priority by scoring partial
  trajectories for test-time beam/frontier control. The Scaling Flaws caution
  requires reversible or fallback pruning, repeated-sampling controls, and
  held-out games for transfer claims.
- **ThinkPRM generative process verifier** (arXiv:2504.16828,
  arXiv:2502.00271): use a long-CoT process verifier only as the sparse,
  expensive check on ambiguous high-upside branches inside Exp 4569. It fails
  if verification cost eats the first-contact action budget or local step
  plausibility is mistaken for final progress.
- **adaptive PRM-guided candidate expansion / best-first scheduling**
  (arXiv:2602.01070,
  arXiv:2502.00271): promote verifier scores from reranking into online budget
  allocation: widen, deepen, prune, or fall back based on process-reward
  aggregates. It fails if it spends more compute without lowering actions to
  first level-up or silently inherits verifier-guided-search scaling flaws.
- **Self-evolving verifiable-reward data refresh** (arXiv:2601.22607,
  arXiv:2502.00271): refresh both the Exp 4568 action-effect prior and Exp 4569
  verifier with executable checked traces and hard negatives from verifier
  pruning failures. It fails if generated traces leak game identity, checkers
  are not execution-tested, or the loop optimizes the verifier rather than
  held-out action efficiency.

## .423 Candidate

flagged_for_v423: use a StochasticGoose-style learned action-effect model as the candidate-expansion prior, then allocate Exp 4569 best-first frontier budget with adaptive PRM guidance and scaling-flaw controls (arXiv:2603.24621 + arXiv:2602.01070 + arXiv:2502.00271)

The practical next experiment should keep the exp4568 positive-control guard,
the exp4569 random-priority/repeated-sampling controls, and the offline
reproduction gate. The implementation change is to move learned clickability
upstream into branch generation, then let adaptive PRM-guided candidate expansion
allocate frontier budget over those branches. AgentRM supplies the
agent-search reward-model precedent, ThinkPRM supplies the expensive
explanation-quality checker for ambiguous branches, self-evolving
verifiable-reward data supplies a refresh stream, and Scaling Flaws supplies the
guardrail: no learned verifier may be the sole irreversible branch killer.
