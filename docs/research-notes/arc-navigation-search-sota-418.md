# ARC navigation-search SOTA ingestion .418 - 2026-06-21

```json
{
  "citations_verified": {
    "1810.02274": {
      "http_status": 200,
      "title": "Episodic Curiosity through Reachability",
      "url": "https://arxiv.org/abs/1810.02274"
    },
    "1901.10995": {
      "http_status": 200,
      "title": "Go-Explore: a New Approach for Hard-Exploration Problems",
      "url": "https://arxiv.org/abs/1901.10995"
    },
    "1906.05253": {
      "http_status": 200,
      "title": "Search on the Replay Buffer: Bridging Planning and Reinforcement Learning",
      "url": "https://arxiv.org/abs/1906.05253"
    },
    "2004.12919": {
      "http_status": 200,
      "title": "First return, then explore",
      "url": "https://arxiv.org/abs/2004.12919"
    },
    "2304.05506": {
      "http_status": 200,
      "title": "Frontier Semantic Exploration for Visual Target Navigation",
      "url": "https://arxiv.org/abs/2304.05506"
    },
    "2602.00460": {
      "http_status": 200,
      "title": "Search Inspired Exploration in Reinforcement Learning",
      "url": "https://arxiv.org/abs/2602.00460"
    },
    "2603.05377": {
      "http_status": 200,
      "title": "OpenFrontier: General Navigation with Visual-Language Grounded Frontiers",
      "url": "https://arxiv.org/abs/2603.05377"
    },
    "2605.25931": {
      "http_status": 200,
      "title": "Explore Before You Solve: The Speed--Depth Trade-off in Epistemic Agents for ARC-AGI-3",
      "url": "https://arxiv.org/abs/2605.25931"
    }
  },
  "field_principles": {
    "citations_verified": "every method claim cites a verifiable arXiv ID/URL -- the no-fabrication bar (same as any results artifact).",
    "field_principles": "principle annotations for every top-level artifact field.",
    "flagged_for_next_roadmap": "the strongest method flagged as a .419 candidate -- closes the discover->ingest->plan loop.",
    "honest_verdict": "terminal prefix; complete: sota_ingestion_navigation_search_mapped.",
    "inference_substrate": "aggregation_from_upstream_artifacts -- literature synthesis, no model load.",
    "methods_mapped": "the 3-5 strongest methods with REAL arXiv IDs -- the shoulders-of-giants anti-rederivation check.",
    "preconditions_checked": "records resources verified; pre-empts missing-resource fabrication.",
    "random_seed": "bare integer seed for reproducible artifact generation.",
    "research_note_path": "repo-relative markdown path for deterministic parsing."
  },
  "flagged_for_next_roadmap": "flagged_for_v419: SoRB-style replay-buffer graph over StepwiseExplorer frontier nodes, with exact _shortest_path navigation costs, charged return prefixes, RESET fallback diagnostics, and the existing CORE median-action gate as the acceptance metric",
  "honest_verdict": "complete: sota_ingestion_navigation_search_mapped",
  "inference_substrate": "aggregation_from_upstream_artifacts",
  "methods_mapped": [
    {
      "fails_when": "the distance metric is learned or stale rather than exact; any unverified edge can turn a RESET-free improvement into a hidden teleport assumption",
      "method": "replay-buffer graph search for physical return paths",
      "source_ids": [
        "1906.05253"
      ],
      "takes_over_current_explorer": ".418 StepwiseExplorer forward-walk navigation fix: promote the visited frame/action-prefix log into a graph, use exact _shortest_path reachability as the edge cost, and choose frontier nodes by navigation cost already payable from the current node",
      "v419_candidate": "flagged_for_v419: SoRB-style replay-buffer graph over StepwiseExplorer nodes with exact _shortest_path costs"
    },
    {
      "fails_when": "episode-start subgoal selection is copied literally; ARC cannot spend extra actions returning to a frontier unless the path is already forward-walkable or replay-accounted",
      "method": "search-inspired reachable-frontier subgoal control",
      "source_ids": [
        "2602.00460",
        "1810.02274"
      ],
      "takes_over_current_explorer": ".418 StepwiseExplorer forward-walk navigation fix: use cost-to-come, cost-to-go, and reachability novelty as equal-depth frontier tie-breaks after depth remains primary",
      "v419_candidate": "flagged_for_v419: SIERL-style frontier score with a reachability novelty guard, never overriding depth"
    },
    {
      "fails_when": "the method relies on emulator state restore, uncharged RESET, or post-hoc robustification training; the live ARC agent must physically navigate to the frontier it probes",
      "method": "Go-Explore archive discipline without state restore",
      "source_ids": [
        "1901.10995",
        "2004.12919"
      ],
      "takes_over_current_explorer": ".418 StepwiseExplorer forward-walk navigation fix: keep the archive of promising states but replace simulator state restore with policy-based or exact replay returns that charge every action",
      "v419_candidate": "flagged_for_v419: Go-Explore archive rows with charged return prefixes and RESET fallback diagnostics"
    },
    {
      "fails_when": "visual-language semantics or dense maps are treated as available inside ARC; only the frontier-cost abstraction transfers cleanly",
      "method": "embodied frontier navigation scoring",
      "source_ids": [
        "2304.05506",
        "2603.05377"
      ],
      "takes_over_current_explorer": ".418 StepwiseExplorer forward-walk navigation fix: score frontier nodes as navigation targets with information gain and reachable-path cost, not just local action priority",
      "v419_candidate": "flagged_for_v419: embodied-navigation frontier score as a diagnostic secondary term behind exact ARC reachability"
    },
    {
      "fails_when": "public-game shortcuts or null-coordinate exploits drive the budget policy; hidden-game action efficiency must be measured by the canonical gate, not inferred from public quirks",
      "method": "ARC speed-depth budget controller",
      "source_ids": [
        "2605.25931"
      ],
      "takes_over_current_explorer": ".418 StepwiseExplorer forward-walk navigation fix: measure whether batching and navigation-cost tie-breaks stay on the action-efficiency frontier instead of only increasing search depth",
      "v419_candidate": "flagged_for_v419: AERA-style speed-depth ledger for every frontier batch and navigation-cost treatment"
    }
  ],
  "preconditions_checked": {
    "agents_md_read": true,
    "arxiv_api_reachable": true,
    "arxiv_http_200_verified_ids": [
      "1906.05253",
      "2602.00460",
      "1901.10995",
      "2004.12919",
      "2605.25931",
      "2304.05506",
      "2603.05377",
      "1810.02274"
    ],
    "codex_md_read": true,
    "deep_research_invoked": false,
    "forward_walk_artifact_read": true,
    "leaderboard_submission": false,
    "live_llm_inference": false,
    "live_solve_claim": false,
    "nav_metric_artifact_read": true,
    "ops_docs_modified": false,
    "prior_action_efficiency_note_read": true,
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
      "RESET-free tree search interactive agents navigation cost frontier",
      "backtracking efficient exploration reinforcement learning frontier navigation cost",
      "replay buffer graph search shortest path reinforcement learning interactive environments",
      "go-explore archive return explore hard exploration no reset",
      "amortized search agents cannot teleport physically navigate"
    ],
    "sweep_semscholar_rate_limited_queries": [
      "backtracking efficient exploration reinforcement learning frontier navigation cost",
      "replay buffer graph search shortest path reinforcement learning interactive environments",
      "go-explore archive return explore hard exploration no reset",
      "amortized search agents cannot teleport physically navigate"
    ],
    "sweep_semscholar_used": true,
    "training_launched": false,
    "websearch_webfetch_top_sources": [
      "https://arxiv.org/abs/1906.05253",
      "https://arxiv.org/abs/2602.00460",
      "https://arxiv.org/abs/1901.10995",
      "https://arxiv.org/abs/2004.12919",
      "https://arxiv.org/abs/2605.25931",
      "https://arxiv.org/abs/2304.05506",
      "https://arxiv.org/abs/2603.05377",
      "https://arxiv.org/abs/1810.02274"
    ]
  },
  "random_seed": 4530,
  "research_note_path": "docs/research-notes/arc-navigation-search-sota-418.md"
}
```

Reliable channel only: `scripts/sweep_clusters.py`, `scripts/sweep_semscholar.py`,
arXiv abs-page HTTP-200 checks, and low-concurrency WebSearch/WebFetch of the
top eight navigation/search sources. Preconditions passed before any claim was
promoted: `.venv/bin/python scripts/sweep_clusters.py --help` exited zero and
`curl -sf -o /dev/null https://export.arxiv.org/api/query?search_query=all:test`
confirmed arXiv API reachability. `scripts/sweep_clusters.py 5 --max-results 8`
and `scripts/sweep_clusters.py 6 --max-results 8` emitted the ARC
action-efficiency and neural-guided-search cluster URLs. Semantic Scholar returned zero unique arXiv IDs
on the focused navigation pass and HTTP 429 on four of five queries, so no
S2-only claim was promoted. No `/deep-research`
call was made. No training, live LLM inference, leaderboard submission, or live
solve was launched. No ops/status/traceability files or
`scripts/research_conductor.py` were modified.

Already-discovered corpus read through an ARC navigation/search filter:
`research-studying.md`, `research-references.md`,
`docs/research-notes/arc-action-efficiency-sota-417.md`,
`results/experiment_4523_forward_walk_navigation.json`, and
`results/experiment_4527_nav_metric_harness.json`. The .418 state this maps
onto is the `StepwiseExplorer` forward-walk navigation fix: depth stays primary,
`_shortest_path` exact reachability can break equal-depth ties, frontier batches
amortize navigation already paid to a node, and RESET replay remains a fallback
diagnostic rather than a free teleport.

Sources checked: arXiv:1906.05253, arXiv:2602.00460, arXiv:1901.10995, arXiv:2004.12919, arXiv:2605.25931, arXiv:2304.05506, arXiv:2603.05377, arXiv:1810.02274.

## Per-Method Mapping

- **Replay-buffer graph search for physical return paths** (arXiv:1906.05253):
  take over the replay/navigation substrate by turning visited frame hashes and
  action prefixes into a graph. The .419 version should use exact `_shortest_path`
  costs instead of a learned distance metric, because the ARC agent cannot
  teleport to a frontier node. This is the strongest backtrack-efficient,
  RESET-free tree-search graft.
- **Search-inspired reachable-frontier subgoal control** (arXiv:2602.00460,
  arXiv:1810.02274): take over equal-depth frontier ordering with cost-to-come,
  cost-to-go, and reachability novelty. It fails if copied as an episode-start
  subgoal policy that spends uncharged return actions.
- **Go-Explore archive discipline without state restore** (arXiv:1901.10995,
  arXiv:2004.12919): preserve the archive-return-explore loop but charge every
  return path through policy-based or exact replay returns. It fails when state
  restore, RESET, or robustification training hides the physical navigation cost.
- **Embodied frontier navigation scoring** (arXiv:2304.05506,
  arXiv:2603.05377): borrow the frontier navigation cost framing from visual
  navigation, but keep only the reachable-frontier abstraction. Language priors
  and dense maps are not available inside ARC.
- **ARC speed-depth budget controller** (arXiv:2605.25931): keep the .418
  frontier-batch and nav-cost sweep honest by asking whether extra search depth
  stays on the action-efficiency frontier. It fails if public-game shortcuts are
  mistaken for hidden-game efficiency.

## bottom line for the .419 roadmap

flagged_for_v419: SoRB-style replay-buffer graph over StepwiseExplorer frontier nodes, with exact _shortest_path navigation costs, charged return prefixes, RESET fallback diagnostics, and the existing CORE median-action gate as the acceptance metric

The method should take over only the navigation layer of the current explorer:
build the replay-buffer graph from real `StepwiseExplorer` nodes, score frontier
targets by exact forward-walk distance first, charge any replay suffix, and keep
the CORE median-action gate from `experiment_4523_forward_walk_navigation.json`.
SIERL and Go-Explore remain supporting controls for frontier priority and archive
discipline. Embodied frontier navigation and AERA are diagnostics for path cost
and speed-depth accounting, not permission to add a new planner that bypasses the
existing submission gate.
