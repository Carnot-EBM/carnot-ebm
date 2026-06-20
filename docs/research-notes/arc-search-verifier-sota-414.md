# ARC search/verifier SOTA ingestion .414 - 2026-06-20

```json
{
  "field_principles": {
    "honest_verdict": "MUST start with a terminal prefix complete:/complete_/success:/success_/passed:/passed_/shipped:/shipped_ so the reconciler classifies it as terminal (Verdict Terminal-Prefix Discipline).",
    "inference_substrate": "explicit declaration (live_llm_inference | verifier_ensemble_against_cached_candidates | aggregation_from_upstream_artifacts) so adversarial_verify applies the right floor.",
    "offline_reproduced": "a solve not reproducible offline is wasted effort -- only reproduced levels count (ARC Solve Reproducibility).",
    "preconditions_checked": "records WHICH resources were verified before launching; pre-empts the silent-missing-resource fabrication mode.",
    "reproduced_levels": "headline metric reproducible_total_levels grows monotonically; report the count banked, real-env-confirmed."
  },
  "gap_mapping": {
    "GAP-ARCH-FEATURES": {
      "next_experiment": "Replace the weak flat hand-feature vector with object-slot, transition-delta, frame-change, and frontier-state features.",
      "principle": "Relational/delta verifier features should describe object slots, frame-change facts, transition deltas, and state-graph context before a candidate is scored.",
      "source_ids": [
        "2606.12316",
        "2512.24156",
        "2605.25931"
      ]
    },
    "GAP-ARCH-GOAL": {
      "next_experiment": "Induce candidate goal predicates separately from executable dynamics, then use verifier-rejected trajectories and disambiguating queries to decide or abstain.",
      "principle": "Goal-vs-dynamics induction must separate what counts as winning from the transition model that predicts how actions change state.",
      "source_ids": [
        "2603.24621",
        "2605.05138",
        "2512.22336",
        "2604.08792"
      ]
    },
    "GAP-ARCH-NO-HIERARCHICAL-SEARCH": {
      "next_experiment": "Add a frontier graph plus MCTS-style verifier-guided expansion over partial plans and candidate world-model edits.",
      "principle": "Hierarchical/MCTS search should expand a verified state-action graph, backpropagate verifier feedback, and spend actions on frontier states rather than flat repeated BFS.",
      "source_ids": [
        "2512.24156",
        "2605.05138",
        "2605.25931",
        "2402.08147"
      ]
    }
  },
  "honest_verdict": "complete: arc_search_verifier_sota_414_mapped_for_v415",
  "inference_substrate": "aggregation_from_upstream_artifacts",
  "methods": [
    {
      "arxiv_id": "2512.24156",
      "mapped_gap": "GAP-ARCH-NO-HIERARCHICAL-SEARCH",
      "name": "Graph-Based Exploration for ARC-AGI-3 Interactive Reasoning Tasks",
      "pitfall": "It is training-free exploration, not a learned verifier; Carnot still needs reproducible env traces before any solve is banked.",
      "stack_mapping": "Use explicit state-action graphs, shortest paths to untested state-action pairs, and salience-prioritized actions as the search baseline that a verifier-routed planner must beat."
    },
    {
      "arxiv_id": "2603.24621",
      "mapped_gap": "GAP-ARCH-GOAL",
      "name": "ARC-AGI-3: A New Challenge for Frontier Agentic Intelligence",
      "pitfall": "The benchmark definition is not a method; it anchors metrics and prevents goal/dynamics claims from being conflated.",
      "stack_mapping": "Treat goal discovery, dynamics modeling, and efficient planning as separate measured axes rather than one opaque route score."
    },
    {
      "arxiv_id": "2605.05138",
      "mapped_gap": "GAP-ARCH-GOAL",
      "name": "Executable World Models for ARC-AGI-3",
      "pitfall": "Published playthroughs are not Carnot evidence; fresh local reproduction and leakage controls remain the only bankable signal.",
      "stack_mapping": "Maintain an executable model, verify it against observations, refactor toward simpler dynamics, and plan through it only after the verifier accepts the predicted transition behavior."
    },
    {
      "arxiv_id": "2606.12316",
      "mapped_gap": "GAP-ARCH-FEATURES",
      "name": "Slots, Transitions, Loops: Learning Composable World Models for ARC",
      "pitfall": "ARC-1/2 grid transitions omit interactive action costs and hidden state, so the features must be checked against live-game deltas.",
      "stack_mapping": "Port object slots, demonstration-conditioned summaries, looped transitions, and correction signals into the verifier feature bank."
    },
    {
      "arxiv_id": "2512.22336",
      "mapped_gap": "GAP-ARCH-GOAL",
      "name": "Agent2World adaptive symbolic world-model feedback",
      "pitfall": "The paper includes a web-searching agent stage; this artifact only takes the behavior-aware testing pattern, not the live research loop.",
      "stack_mapping": "Use adaptive unit tests and simulation-based validation to expose behavior-level world-model errors before the planner trusts a rule."
    },
    {
      "arxiv_id": "2605.25931",
      "mapped_gap": "GAP-ARCH-NO-HIERARCHICAL-SEARCH",
      "name": "AERA speed-depth explore/verify/plan framework",
      "pitfall": "The public-set vulnerability means public scores cannot be promoted without private-style or hidden-state robustness checks.",
      "stack_mapping": "Budget exploration for information gain, then verify and plan; use its benchmark critique as a guard against public-set shortcuts."
    },
    {
      "arxiv_id": "2604.08792",
      "mapped_gap": "GAP-ARCH-GOAL",
      "name": "Choose, Don't Label program-disambiguation queries",
      "pitfall": "The original uses humans for intent answers; Carnot must replace that answerer with executable evidence, or mark underdetermined.",
      "stack_mapping": "When candidate dynamics agree on demos but imply different goals, synthesize a discriminating behavior query and require replayable evidence to choose or abstain."
    },
    {
      "arxiv_id": "2402.08147",
      "mapped_gap": "GAP-ARCH-NO-HIERARCHICAL-SEARCH",
      "name": "VerMCTS verifier-guided tree search",
      "pitfall": "Dafny/Coq verified code is not ARC control; the transferable part is verifier-in-the-loop tree search, not the proof benchmark.",
      "stack_mapping": "Adapt verifier-scored partial-program MCTS to partial world-model edits and plan prefixes, using verifier failures to avoid doomed branches early."
    }
  ],
  "offline_reproduced": false,
  "preconditions_checked": {
    "agents_md_read": true,
    "arxiv_http_200_verified_ids": [
      "2512.24156",
      "2603.24621",
      "2605.05138",
      "2606.12316",
      "2512.22336",
      "2605.25931",
      "2604.08792",
      "2402.08147"
    ],
    "codex_md_read": true,
    "deep_research_invoked": false,
    "leaderboard_submission": false,
    "live_llm_inference": false,
    "live_solve_claim": false,
    "ops_docs_modified": false,
    "research_references_filtered": true,
    "research_studying_filtered": true,
    "sweep_clusters_help_succeeded": true,
    "sweep_clusters_urls": [
      "http://export.arxiv.org/api/query?search_query=(abs:\"verifier+ensemble\"+OR+abs:\"verifier+ensembles\"+OR+abs:\"null+space\"+OR+abs:\"specification+gaming\"+OR+abs:\"process+reward+model\"+OR+abs:\"deliberative+alignment\"+OR+abs:\"reward+hacking\")&start=0&max_results=8&sortBy=submittedDate&sortOrder=descending",
      "http://export.arxiv.org/api/query?search_query=(abs:\"active+inference\"+OR+abs:\"free+energy\"+OR+abs:\"free+energy+principle\"+OR+abs:\"predictive+coding\"+OR+abs:\"world+model\")+AND+(abs:\"LLM\"+OR+abs:\"language+model\"+OR+abs:\"reasoning\")&start=0&max_results=8&sortBy=submittedDate&sortOrder=descending"
    ],
    "training_launched": false,
    "websearch_webfetch_top_sources": [
      "https://arxiv.org/abs/2512.24156",
      "https://arxiv.org/abs/2603.24621",
      "https://arxiv.org/abs/2605.05138",
      "https://arxiv.org/abs/2606.12316",
      "https://arxiv.org/abs/2512.22336",
      "https://arxiv.org/abs/2605.25931",
      "https://arxiv.org/abs/2604.08792",
      "https://arxiv.org/abs/2402.08147"
    ]
  },
  "random_seed": 4478,
  "reproduced_levels": 0,
  "research_note_path": "docs/research-notes/arc-search-verifier-sota-414.md",
  "source_ids": [
    "2512.24156",
    "2603.24621",
    "2605.05138",
    "2606.12316",
    "2512.22336",
    "2605.25931",
    "2604.08792",
    "2402.08147"
  ],
  "strongest_for_v415": "flagged_for_v415: graph-state/delta-feature verifier plus hierarchical verified search, anchored by arXiv:2512.24156, arXiv:2606.12316, and arXiv:2402.08147"
}
```

Reliable channel only: `research-studying.md`, `research-references.md`,
`scripts/sweep_clusters.py`, arXiv abs-page HTTP-200 checks, and
low-concurrency WebSearch/WebFetch of the top eight ARC search/verifier
sources. `.venv/bin/python scripts/sweep_clusters.py --help` succeeded.
`scripts/sweep_clusters.py 0 --max-results 8` and
`scripts/sweep_clusters.py 3 --max-results 8` emitted verifier and world-model
cluster URLs. No `/deep-research` call was made. No live solve, live LLM
inference, training run, or leaderboard submission was launched.

Sources checked: arXiv:2512.24156, arXiv:2603.24621, arXiv:2605.05138, arXiv:2606.12316, arXiv:2512.22336, arXiv:2605.25931, arXiv:2604.08792, arXiv:2402.08147.

## Gap Mapping

- GAP-ARCH-FEATURES: relational/delta verifier features should combine
  object slots, frame-change facts, transition deltas, and local state-graph
  context before scoring candidate plans.
- GAP-ARCH-GOAL: goal-vs-dynamics induction should infer the win predicate
  separately from the transition model, then use replayable counterexamples or
  disambiguating behavior queries when candidates disagree.
- GAP-ARCH-NO-HIERARCHICAL-SEARCH: hierarchical/MCTS verifier-guided search
  should replace flat repeated BFS with state-graph frontier expansion and
  verifier feedback over partial plans or world-model edits.

## Focused Sweep Result

- Graph-Based Exploration for ARC-AGI-3, arXiv:2512.24156, is the strongest
  direct search baseline: explicit state-action graph, salience-prioritized
  actions, and shortest paths to untested state-action pairs.
- ARC-AGI-3 benchmark, arXiv:2603.24621, is the metric anchor: agents must
  explore, infer goals, build dynamics, and plan efficiently without language
  instructions.
- Executable World Models, arXiv:2605.05138, is the verifier-grounded dynamics
  substrate: maintain executable model, verify against observations, simplify,
  and plan through it.
- Loop-OWM, arXiv:2606.12316, is the best fit for relational/delta state
  features: slots, transitions, loops, dense propagation, and correction.
- Agent2World, arXiv:2512.22336, supplies adaptive behavior-level tests for
  executable symbolic world models.
- AERA, arXiv:2605.25931, sharpens the explore/verify/plan action budget and
  warns that public ARC-AGI-3 scores can be shortcut by trivial strategies.
- Choose, Don't Label, arXiv:2604.08792, supplies the disambiguating query
  pattern for underdetermined candidate goals or dynamics.
- VerMCTS, arXiv:2402.08147, supplies verifier-in-the-loop tree search over
  partial programs; the transferable part is MCTS-style expansion with cheap
  verifier rejection.

## SOTA->Experiment Mapping

For `.415`, build the graph-state/delta-feature verifier plus hierarchical
verified search package: a state-action graph feeds relational/delta features
into the verifier, candidate goals are induced separately from executable
dynamics, and MCTS-style expansion uses verifier feedback to prune partial
plans or world-model edits. Count this artifact as a planning hand-off only:
`offline_reproduced=false`, `reproduced_levels=0`, and
`inference_substrate=aggregation_from_upstream_artifacts`.

flagged_for_v415: graph-state/delta-feature verifier plus hierarchical verified search, anchored by arXiv:2512.24156, arXiv:2606.12316, and arXiv:2402.08147
