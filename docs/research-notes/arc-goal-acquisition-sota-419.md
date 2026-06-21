# ARC goal-acquisition SOTA ingestion .419 - 2026-06-21

```json
{
  "citations_verified": {
    "2310.19791": {
      "http_status": 200,
      "title": "LILO: Learning Interpretable Libraries by Compressing and Documenting Code",
      "url": "https://arxiv.org/abs/2310.19791"
    },
    "2411.17708": {
      "http_status": 200,
      "title": "Towards Efficient Neurally-Guided Program Induction for ARC-AGI",
      "url": "https://arxiv.org/abs/2411.17708"
    },
    "2507.14172": {
      "http_status": 200,
      "title": "Self-Improving Language Models for Evolutionary Program Synthesis: A Case Study on ARC-AGI",
      "url": "https://arxiv.org/abs/2507.14172"
    },
    "2512.22336": {
      "http_status": 200,
      "title": "Agent2World: Learning to Generate Symbolic World Models via Adaptive Multi-Agent Feedback",
      "url": "https://arxiv.org/abs/2512.22336"
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
    "2604.08792": {
      "http_status": 200,
      "title": "Choose, Don't Label: Multiple-Choice Query Synthesis for Program Disambiguation",
      "url": "https://arxiv.org/abs/2604.08792"
    },
    "2605.05138": {
      "http_status": 200,
      "title": "Executable World Models for ARC-AGI-3 in the Era of Coding Agents",
      "url": "https://arxiv.org/abs/2605.05138"
    }
  },
  "field_principles": {
    "citations_verified": "every method claim cites a verifiable arXiv ID/URL -- the no-fabrication bar (same as any results artifact).",
    "field_principles": "principle annotations for every top-level artifact field.",
    "flagged_for_next_roadmap": "the strongest method flagged as a .420 candidate -- closes the discover->ingest->plan loop.",
    "honest_verdict": "terminal prefix; complete: sota_ingestion_goal_acquisition_mapped.",
    "inference_substrate": "aggregation_from_upstream_artifacts -- literature synthesis, no model load.",
    "methods_mapped": "the 3-5 strongest methods with REAL arXiv IDs -- the shoulders-of-giants anti-rederivation check.",
    "preconditions_checked": "records resources verified; pre-empts missing-resource fabrication.",
    "random_seed": "bare integer seed for reproducible artifact generation.",
    "research_note_path": "repo-relative markdown path for deterministic parsing."
  },
  "flagged_for_next_roadmap": "flagged_for_v420: Family-B executable re-induction loop for each level-up, with separate GOAL-vs-dynamics candidates, adaptive behavior tests for goal-shift detection, and a bounded refinement loop around exp4533",
  "honest_verdict": "complete: sota_ingestion_goal_acquisition_mapped",
  "inference_substrate": "aggregation_from_upstream_artifacts",
  "methods_mapped": [
    {
      "fails_when": "the executable model is treated as proof without held-out transition checks, or when the induced simulator explains L1 dynamics but keeps the stale L1 completion predicate after the episode shifts to L2.",
      "method": "Family-B executable world-model re-induction",
      "source_ids": [
        "2605.05138",
        "2603.24621"
      ],
      "takes_over_current_reinduction": "Exp 4533 currently clears stale induction state after a level-up and asks the offline DSL path for a new level-conditioned GOAL predicate. This method takes over that post-level-up induction slot with a verifier-driven executable Python world model: induce dynamics and GOAL separately, verify predicted transitions against post-transition frames, refactor toward simpler state variables, then route search with the new predicate.",
      "v420_candidate": "flagged_for_v420: Family-B executable re-induction loop for each level-up, with separate GOAL and transition checks"
    },
    {
      "fails_when": "the loop optimizes static ARC-AGI grid transforms rather than interactive ARC-AGI-3 state/action traces, or when hindsight fine-tuning is assumed available inside the autonomous sprint.",
      "method": "refinement-loop program synthesis over candidate GOAL predicates",
      "source_ids": [
        "2601.10904",
        "2507.14172"
      ],
      "takes_over_current_reinduction": "Exp 4533 uses a single deterministic re-induction pass per level-up. This method turns that pass into a bounded refinement loop: generate several candidate GOAL/dynamics programs, execute them on post-transition observations, keep counterexample failures as feedback, and re-synthesize before the next frontier batch.",
      "v420_candidate": "flagged_for_v420: bounded ARC Prize/SOAR-style refinement loop around exp4533 candidate GOAL programs"
    },
    {
      "fails_when": "the test generator depends on web-search agents, human intent answers, or labels unavailable to the offline ARC agent; the replacement answerer must be executable evidence from frames.",
      "method": "adaptive behavior-test goal-shift detector",
      "source_ids": [
        "2512.22336",
        "2604.08792"
      ],
      "takes_over_current_reinduction": "Exp 4533 detects only the level counter increment. This method adds intra-episode goal-shift detection by synthesizing behavior tests that distinguish stale and new GOAL candidates; when tests disagree, the route abstains or re-induces instead of continuing with the old predicate.",
      "v420_candidate": "flagged_for_v420: adaptive behavior-test harness for detecting within-episode GOAL shifts after level-up"
    },
    {
      "fails_when": "library compression memorizes game-specific coordinates or L1 surface predicates; every reused primitive still needs representation-correct post-level-up validation.",
      "method": "neural-guided DSL/library induction for reusable level predicates",
      "source_ids": [
        "2411.17708",
        "2310.19791"
      ],
      "takes_over_current_reinduction": "Exp 4533 re-induces from the current episode in isolation. This method keeps the re-induction trigger but changes the search space: retrieve documented predicate/world-model primitives from the solved ARC corpus, then neurally order compact DSL candidates for the new level before falling back to blind enumeration.",
      "v420_candidate": "flagged_for_v420: LILO/neural-guided predicate library routed by the exp4533 level-up trigger"
    }
  ],
  "preconditions_checked": {
    "agents_md_read": true,
    "arxiv_api_reachable": true,
    "arxiv_http_200_verified_ids": [
      "2603.24621",
      "2605.05138",
      "2601.10904",
      "2507.14172",
      "2512.22336",
      "2604.08792",
      "2411.17708",
      "2310.19791"
    ],
    "codex_md_read": true,
    "deep_research_invoked": false,
    "leaderboard_submission": false,
    "live_llm_inference": false,
    "live_solve_claim": false,
    "ops_docs_modified": false,
    "prior_reinduction_artifact_read": true,
    "prior_reinduction_spec_read": true,
    "research_conductor_modified": false,
    "research_references_filtered": true,
    "research_studying_filtered": true,
    "research_studying_updated": true,
    "superseded_navigation_reingested": false,
    "sweep_clusters_help_exit_0": true,
    "sweep_clusters_urls": [
      "http://export.arxiv.org/api/query?search_query=(abs:\"neural+guided+search\"+OR+abs:\"learned+heuristic\"+OR+abs:\"value+guided+search\"+OR+abs:\"program+induction\"+OR+abs:\"world+model\"+OR+abs:\"goal+induction\")+AND+(abs:\"planning\"+OR+abs:\"agent\"+OR+abs:\"reasoning\"+OR+abs:\"reinforcement+learning\")&start=0&max_results=8&sortBy=submittedDate&sortOrder=descending",
      "http://export.arxiv.org/api/query?search_query=(abs:\"active+inference\"+OR+abs:\"free+energy\"+OR+abs:\"free+energy+principle\"+OR+abs:\"predictive+coding\"+OR+abs:\"world+model\")+AND+(abs:\"LLM\"+OR+abs:\"language+model\"+OR+abs:\"reasoning\")&start=0&max_results=8&sortBy=submittedDate&sortOrder=descending"
    ],
    "sweep_clusters_used": true,
    "sweep_semscholar_arxiv_ids": [
      "2507.14172",
      "2603.20334",
      "2603.13372",
      "2601.10904"
    ],
    "sweep_semscholar_queries": [
      "ARC-AGI-3 goal acquisition executable world models program induction",
      "ARC AGI goal induction program synthesis refinement loop",
      "interactive agents goal-shift detection world model induction",
      "ARC Prize 2025 program synthesis refinement loop ARC-AGI",
      "executable world model goal acquisition ARC-AGI-3 Family-B"
    ],
    "sweep_semscholar_rate_limited_queries": [
      "ARC-AGI-3 goal acquisition executable world models program induction",
      "interactive agents goal-shift detection world model induction",
      "ARC Prize 2025 program synthesis refinement loop ARC-AGI"
    ],
    "sweep_semscholar_used": true,
    "training_launched": false,
    "websearch_webfetch_top_sources": [
      "https://arxiv.org/abs/2603.24621",
      "https://arxiv.org/abs/2605.05138",
      "https://arxiv.org/abs/2601.10904",
      "https://arxiv.org/abs/2507.14172",
      "https://arxiv.org/abs/2512.22336",
      "https://arxiv.org/abs/2604.08792",
      "https://arxiv.org/abs/2411.17708",
      "https://arxiv.org/abs/2310.19791"
    ]
  },
  "random_seed": 4541,
  "research_note_path": "docs/research-notes/arc-goal-acquisition-sota-419.md"
}
```

Reliable channel only: `scripts/sweep_clusters.py`, `scripts/sweep_semscholar.py`,
arXiv abs-page HTTP-200 checks, and low-concurrency WebSearch/WebFetch of the
top eight goal-acquisition and world-model sources. Preconditions passed before
any claim was promoted: `.venv/bin/python scripts/sweep_clusters.py --help`
exited zero and
`curl -sf -o /dev/null https://export.arxiv.org/api/query?search_query=all:test`
confirmed arXiv API reachability. `scripts/sweep_clusters.py 6 --max-results 8`
and `scripts/sweep_clusters.py 3 --max-results 8` emitted the goal/world-model
cluster URLs. `scripts/sweep_semscholar.py` returned arXiv:2507.14172,
arXiv:2603.20334, arXiv:2603.13372, and arXiv:2601.10904, with HTTP 429 on
three focused queries, so no S2-only claim was promoted. No `/deep-research`
call was made. No training, live LLM inference, leaderboard submission, or live
solve was launched. No ops/status/traceability files or
`scripts/research_conductor.py` were modified, and the navigation thread is superseded
rather than re-ingested.

Already-discovered corpus read through an ARC goal-acquisition / world-model
induction filter: `research-studying.md`, `research-references.md`,
`openspec/capabilities/arc-world-model-trust-energy/spec.md` at
`REQ-ARC-WMTE-4533`, and
`results/experiment_4533_per_level_goal_reinduction.json`. The current
mechanism this maps onto is exp4533: after a level-up it clears stale induction
state, re-runs post-transition induction, and biases depth-primary frontier
search with a new level-conditioned GOAL predicate. The .419 headline is
reaching deeper levels through per-level / intra-episode GOAL induction,
goal-shift detection, Family-B executable world-model induction, and
refinement-loop program synthesis.

Sources checked: arXiv:2603.24621, arXiv:2605.05138, arXiv:2601.10904, arXiv:2507.14172, arXiv:2512.22336, arXiv:2604.08792, arXiv:2411.17708, arXiv:2310.19791.

## Per-Method Mapping

- **Family-B executable world-model re-induction** (arXiv:2605.05138,
  arXiv:2603.24621): replace the single exp4533 offline-DSL predicate pass
  with a verifier-driven loop that induces GOAL and transition candidates
  separately, checks post-transition held-out transitions, refactors state, and
  plans only through accepted executable models. This is the strongest .420
  candidate because it is the closest direct fit to ARC-AGI-3 goal acquisition.
- **Refinement-loop program synthesis over candidate GOAL predicates**
  (arXiv:2601.10904, arXiv:2507.14172): make the post-level-up induction pass
  iterative. Failed candidates become execution counterexamples for a bounded
  re-synthesis loop before the next frontier batch. It fails if imported as a
  static ARC-AGI grid-transform solver without interactive action traces.
- **Adaptive behavior-test goal-shift detector** (arXiv:2512.22336,
  arXiv:2604.08792): synthesize behavior tests that distinguish stale and new
  GOAL candidates after a level-up. It fails if the test answerer is a human or
  web-search agent rather than executable frame evidence.
- **Neural-guided DSL/library induction for reusable level predicates**
  (arXiv:2411.17708, arXiv:2310.19791): retrieve documented primitives and
  neurally order compact DSL candidates after the exp4533 trigger. It fails
  when compressed libraries memorize coordinates or old L1 predicates instead
  of representation-correct post-level-up rules.

## bottom line for the .420 roadmap

flagged_for_v420: Family-B executable re-induction loop for each level-up, with separate GOAL-vs-dynamics candidates, adaptive behavior tests for goal-shift detection, and a bounded refinement loop around exp4533

The practical next experiment should keep exp4533's level-up trigger and
depth-primary route, but replace the one-shot predicate induction body with a
small Family-B executable world-model loop. GOAL-vs-dynamics separation is the
first check. Adaptive behavior tests should detect a within-episode goal-shift
before search spends another batch under a stale predicate. Refinement-loop
program synthesis should be bounded and execution-grounded; no live LLM load,
training run, leaderboard submission, or new solve claim is implied by this
ingestion artifact.
