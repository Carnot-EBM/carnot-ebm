# ARC action-efficiency SOTA ingestion .417 - 2026-06-20

```json
{
  "citations": {
    "1511.05952": {
      "http_status": 200,
      "title": "Prioritized Experience Replay",
      "url": "https://arxiv.org/abs/1511.05952"
    },
    "1704.03732": {
      "http_status": 200,
      "title": "Deep Q-learning from Demonstrations",
      "url": "https://arxiv.org/abs/1704.03732"
    },
    "1901.10995": {
      "http_status": 200,
      "title": "Go-Explore: a New Approach for Hard-Exploration Problems",
      "url": "https://arxiv.org/abs/1901.10995"
    },
    "2008.09241": {
      "http_status": 200,
      "title": "Learning Affordance Landscapes for Interaction Exploration in 3D Environments",
      "url": "https://arxiv.org/abs/2008.09241"
    },
    "2501.06047": {
      "http_status": 200,
      "title": "Learning Affordances from Interactive Exploration using an Object-level Map",
      "url": "https://arxiv.org/abs/2501.06047"
    },
    "2602.00460": {
      "http_status": 200,
      "title": "Search Inspired Exploration in Reinforcement Learning",
      "url": "https://arxiv.org/abs/2602.00460"
    },
    "2602.03201": {
      "http_status": 200,
      "title": "SLOPE: Optimistic Potential Landscape Shaping for Model-based Reinforcement Learning",
      "url": "https://arxiv.org/abs/2602.03201"
    },
    "2602.05832": {
      "http_status": 200,
      "title": "UI-Mem: Self-Evolving Experience Memory for Online Reinforcement Learning in Mobile GUI Agents",
      "url": "https://arxiv.org/abs/2602.05832"
    }
  },
  "field_principles": {
    "citations": "real arXiv IDs / URLs for every method claim (the two-source / pre-claim checklist).",
    "field_principles": "principle annotations for every top-level artifact field.",
    "honest_verdict": "terminal prefix; e.g. complete: action_efficiency_sota_417_mapped_for_v418.",
    "inference_substrate": "aggregation_from_upstream_artifacts -- literature synthesis, no compute (100us floor).",
    "methods_mapped": "the strongest 3-5 methods with real arXiv IDs -- a claim without a verifiable citation is fabrication.",
    "preconditions_checked": "records network was verified; pre-empts fabricated-citation failure.",
    "random_seed": "bare integer seed for reproducible artifact generation.",
    "research_note_path": "repo-relative markdown path for deterministic parsing.",
    "source_ids": "five to eight real arXiv IDs promoted by the reliable channel.",
    "v418_flagged_candidates": "closes the discover->ingest->plan loop so SOTA flows into .418 experiments."
  },
  "honest_verdict": "complete: action_efficiency_sota_417_mapped_for_v418",
  "inference_substrate": "aggregation_from_upstream_artifacts",
  "methods_mapped": [
    {
      "fails_when": "availability is mistaken for progress; a clickable cell can still be a loop unless the lazy value/energy term verifies movement toward level progress",
      "method": "affordance-landscape clickability pruning",
      "source_ids": [
        "2008.09241",
        "2501.06047"
      ],
      "takes_over_current_stack": "offline-search + lazy value head + frame-change predictor: replace blind candidate expansion with a frame-only click/action affordance mask before BFS spends actions",
      "v418_candidate": "flagged_for_v418: train the frame-change predictor as an affordance heatmap and prune predicted no-op action/click cells"
    },
    {
      "fails_when": "frontier bookkeeping becomes a second solver with hand-tuned cell abstractions; ARC acceptance must stay actions-to-first-levelup at equal solve-rate",
      "method": "search-inspired frontier control",
      "source_ids": [
        "2602.00460",
        "1901.10995"
      ],
      "takes_over_current_stack": "offline-search + lazy value head + frame-change predictor: choose frontier state-action pairs that are reachable, not exhausted, and promising under lazy value instead of repeatedly expanding flat BFS",
      "v418_candidate": "flagged_for_v418: add a SIERL/Go-Explore frontier queue over cached frame hashes and replayable action prefixes"
    },
    {
      "fails_when": "public-game demonstrations dominate hidden-game behavior; priorities must decay unless held-out variants show equal or better progress",
      "method": "prioritized replay with demonstration seeding",
      "source_ids": [
        "1511.05952",
        "1704.03732"
      ],
      "takes_over_current_stack": "offline-search + lazy value head + frame-change predictor: sample rare progress, human replay, and high-TD-error transitions before uniform self-play when training the predictor and lazy value head",
      "v418_candidate": "flagged_for_v418: seed predictor/value batches with PER/DQfD-style expert transitions, then anneal after self-play catches up"
    },
    {
      "fails_when": "retrieval is not gated by semantic/frame similarity; irrelevant memory can waste actions faster than blind search",
      "method": "persistent hierarchical action memory",
      "source_ids": [
        "2602.05832"
      ],
      "takes_over_current_stack": "offline-search + lazy value head + frame-change predictor: persist action-effect templates, failure cautions, and successful prefixes across games as retrieval hints rather than relearning from scratch",
      "v418_candidate": "flagged_for_v418: create a PersistentAEM-style store of frame-diff/action/reward templates with caution suppression"
    },
    {
      "fails_when": "the potential is learned from frame marginals or dense proxy rewards without structural checks; it can over-rank visually novel dead ends",
      "method": "optimistic potential shaping for sparse progress",
      "source_ids": [
        "2602.03201"
      ],
      "takes_over_current_stack": "offline-search + lazy value head + frame-change predictor: use an optimistic potential term beside lazy value so rare level-progress signals rank survivors after no-op pruning",
      "v418_candidate": "flagged_for_v418: add SLOPE-style upper-bound progress potential as a ranking-only feature after frame-change pruning"
    }
  ],
  "preconditions_checked": {
    "agents_md_read": true,
    "arxiv_http_200_verified_ids": [
      "2008.09241",
      "2501.06047",
      "2602.00460",
      "2602.03201",
      "1511.05952",
      "1704.03732",
      "1901.10995",
      "2602.05832"
    ],
    "codex_md_read": true,
    "deep_research_invoked": false,
    "input_notes_read": [
      "docs/research-notes/arc-leaderboard-competitive-intel-2026-06-20.md",
      "docs/research-notes/arc-417-shaping-action-efficiency.md",
      "research-studying.md",
      "research-references.md",
      "docs/research-notes/arc-imitation-sota-415.md"
    ],
    "leaderboard_submission": false,
    "live_llm_inference": false,
    "live_solve_claim": false,
    "network_precondition_hf_models_exit_0": true,
    "ops_docs_modified": false,
    "research_conductor_modified": false,
    "research_references_filtered": true,
    "research_studying_filtered": true,
    "research_studying_updated": true,
    "sweep_clusters_urls": [
      "http://export.arxiv.org/api/query?search_query=(abs:\"affordance\"+OR+abs:\"action+effect\"+OR+abs:\"clickability\"+OR+abs:\"frame+prediction\"+OR+abs:\"intrinsic+motivation\"+OR+abs:\"directed+exploration\"+OR+abs:\"novelty+search\")+AND+(abs:\"reinforcement+learning\"+OR+abs:\"agent\"+OR+abs:\"exploration\"+OR+abs:\"interactive+environment\"+OR+abs:\"ARC\")&start=0&max_results=8&sortBy=submittedDate&sortOrder=descending",
      "http://export.arxiv.org/api/query?search_query=(abs:\"neural+guided+search\"+OR+abs:\"learned+heuristic\"+OR+abs:\"value+guided+search\"+OR+abs:\"program+induction\"+OR+abs:\"world+model\"+OR+abs:\"goal+induction\")+AND+(abs:\"planning\"+OR+abs:\"agent\"+OR+abs:\"reasoning\"+OR+abs:\"reinforcement+learning\")&start=0&max_results=8&sortBy=submittedDate&sortOrder=descending"
    ],
    "sweep_clusters_used": true,
    "sweep_semscholar_arxiv_ids": [
      "2509.10511",
      "2210.07553",
      "2601.08665",
      "2507.07451",
      "2511.03405",
      "2402.18487",
      "2207.07791"
    ],
    "sweep_semscholar_queries": [
      "action effect model affordance exploration reinforcement learning persistent memory",
      "experience replay for search action efficiency exploration reinforcement learning",
      "persistent action memory action effect reinforcement learning exploration",
      "action effect prediction reinforcement learning exploration affordance",
      "clickability visual affordance reinforcement learning interactive exploration"
    ],
    "sweep_semscholar_rate_limited_queries": [
      "experience replay for search action efficiency exploration reinforcement learning",
      "persistent action memory action effect reinforcement learning exploration",
      "action effect prediction reinforcement learning exploration affordance",
      "clickability visual affordance reinforcement learning interactive exploration"
    ],
    "sweep_semscholar_used": true,
    "training_launched": false,
    "websearch_webfetch_top_sources": [
      "https://arxiv.org/abs/2008.09241",
      "https://arxiv.org/abs/2501.06047",
      "https://arxiv.org/abs/2602.00460",
      "https://arxiv.org/abs/2602.03201",
      "https://arxiv.org/abs/1511.05952",
      "https://arxiv.org/abs/1704.03732",
      "https://arxiv.org/abs/1901.10995",
      "https://arxiv.org/abs/2602.05832"
    ]
  },
  "random_seed": 4520,
  "research_note_path": "docs/research-notes/arc-action-efficiency-sota-417.md",
  "source_ids": [
    "2008.09241",
    "2501.06047",
    "2602.00460",
    "2602.03201",
    "1511.05952",
    "1704.03732",
    "1901.10995",
    "2602.05832"
  ],
  "v418_flagged_candidates": [
    "flagged_for_v418: affordance-pruned frame-change/clickability model anchored by arXiv:2008.09241 and arXiv:2501.06047",
    "flagged_for_v418: SIERL/Go-Explore frontier queue over replayable offline-search states anchored by arXiv:2602.00460 and arXiv:1901.10995",
    "flagged_for_v418: PER/DQfD transition sampler for the frame-change predictor and lazy value head anchored by arXiv:1511.05952 and arXiv:1704.03732",
    "flagged_for_v418: UI-Mem-style persistent cross-game action memory with caution suppression anchored by arXiv:2602.05832",
    "flagged_for_v418: SLOPE-style optimistic potential ranking after no-op pruning anchored by arXiv:2602.03201"
  ]
}
```

Reliable channel only: `scripts/sweep_clusters.py`, `scripts/sweep_semscholar.py`,
arXiv abs-page HTTP-200 checks, and low-concurrency WebSearch/WebFetch of the
top eight action-efficient exploration sources. The network precondition
`curl -sf -o /dev/null https://huggingface.co/api/models` succeeded before any
claim was promoted. `scripts/sweep_clusters.py 5 --max-results 8` and
`scripts/sweep_clusters.py 6 --max-results 8` emitted the action-efficiency and
neural-guided-search cluster URLs. Semantic Scholar returned seven candidate
arXiv IDs across focused queries and reported HTTP 429 on the replay/memory
queries, so no S2-only claim was promoted. No `/deep-research` call was made.
No training, live LLM inference, leaderboard submission, or live solve was
launched. No ops/status/traceability files or `scripts/research_conductor.py`
were modified.

Sources checked: arXiv:2008.09241, arXiv:2501.06047, arXiv:2602.00460, arXiv:2602.03201, arXiv:1511.05952, arXiv:1704.03732, arXiv:1901.10995, arXiv:2602.05832.

## Per-Method Mapping

- **Affordance-landscape clickability pruning** (arXiv:2008.09241,
  arXiv:2501.06047): take over the front of the explorer by predicting which
  action/click cells are feasible and likely to change the frame, then prune
  predicted no-ops before BFS. This is the cleanest action-effect/clickability
  graft onto the offline-search + lazy value head + frame-change predictor
  stack. It fails when availability is treated as progress; every survivor
  still needs value/energy or level-progress checks.
- **Search-inspired frontier control** (arXiv:2602.00460, arXiv:1901.10995):
  replace flat breadth expansion with a frontier queue over reachable cached
  frame hashes and replayable prefixes. It fails when the frontier abstraction
  becomes benchmark-specific or loses the action-count acceptance gate.
- **Prioritized replay with demonstration seeding** (arXiv:1511.05952,
  arXiv:1704.03732): train the frame-change predictor and lazy value head from
  high-progress, high-error, and human/demo transitions before uniform self-play.
  It fails when public-game demonstrations remain over-prioritized after
  held-out variants stop improving.
- **Persistent hierarchical action memory** (arXiv:2602.05832): persist
  successful prefixes, frame-diff/action/reward templates, and failure cautions
  across games. It fails when retrieval is not gated by similarity and stale
  memory wastes actions.
- **Optimistic potential shaping** (arXiv:2602.03201): add a ranking-only
  potential term after frame-change pruning so rare sparse-progress signals are
  not flattened. It fails when a proxy potential over-ranks visually novel dead
  ends without structural checks.

## bottom line for the .418 roadmap

The strongest `.418` input is the combined **affordance-pruned frame-change
predictor plus frontier control**: prune predicted no-ops first, then let a
SIERL/Go-Explore-style frontier queue choose among reachable survivors under
the lazy value head. PER/DQfD replay seeding is the training substrate for that
predictor/value pair. UI-Mem-style persistent action memory is second-line:
use it only with similarity-gated retrieval and caution suppression. SLOPE-style
potential shaping is ranking-only until it proves actions-to-first-levelup
improvement at equal solve-rate.

flagged_for_v418: affordance-pruned frame-change/clickability model anchored by arXiv:2008.09241 and arXiv:2501.06047
flagged_for_v418: SIERL/Go-Explore frontier queue over replayable offline-search states anchored by arXiv:2602.00460 and arXiv:1901.10995
flagged_for_v418: PER/DQfD transition sampler for the frame-change predictor and lazy value head anchored by arXiv:1511.05952 and arXiv:1704.03732
flagged_for_v418: UI-Mem-style persistent cross-game action memory with caution suppression anchored by arXiv:2602.05832
flagged_for_v418: SLOPE-style optimistic potential ranking after no-op pruning anchored by arXiv:2602.03201
