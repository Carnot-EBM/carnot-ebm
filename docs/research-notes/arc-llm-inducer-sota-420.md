# ARC LLM-inducer SOTA ingestion .420 - 2026-06-21

```json
{
  "citations_verified": {
    "2305.14591": {
      "http_status": 200,
      "title": "ALGO: Synthesizing Algorithmic Programs with LLM-Generated Oracle Verifiers",
      "url": "https://arxiv.org/abs/2305.14591"
    },
    "2510.14331": {
      "http_status": 200,
      "title": "LLM Priors for ERM over Programs",
      "url": "https://arxiv.org/abs/2510.14331"
    },
    "2603.20334": {
      "http_status": 200,
      "title": "Procedural Refinement by LLM-driven Algorithmic Debugging for ARC-AGI-2",
      "url": "https://arxiv.org/abs/2603.20334"
    },
    "2603.24621": {
      "http_status": 200,
      "title": "ARC-AGI-3: A New Challenge for Frontier Agentic Intelligence",
      "url": "https://arxiv.org/abs/2603.24621"
    },
    "2605.05138": {
      "http_status": 200,
      "title": "Executable World Models for ARC-AGI-3 in the Era of Coding Agents",
      "url": "https://arxiv.org/abs/2605.05138"
    },
    "2606.11521": {
      "http_status": 200,
      "title": "Counterexample Guided Learning in the Large using Reasoning Agents",
      "url": "https://arxiv.org/abs/2606.11521"
    }
  },
  "field_principles": {
    "citations_verified": "every method claim cites a verifiable arXiv ID/URL -- the no-fabrication bar (same as any results artifact).",
    "field_principles": "principle annotations for every top-level artifact field.",
    "flagged_for_next_roadmap": "the strongest method flagged as a .421 candidate -- closes the discover->ingest->plan loop.",
    "honest_verdict": "terminal prefix; complete: sota_ingestion_llm_inducer_mapped.",
    "inference_substrate": "aggregation_from_upstream_artifacts -- literature synthesis, no model load.",
    "methods_mapped": "the 3-5 strongest methods with REAL arXiv IDs -- the shoulders-of-giants anti-rederivation check.",
    "preconditions_checked": "records resources verified; pre-empts missing-resource fabrication.",
    "random_seed": "bare integer seed for reproducible artifact generation.",
    "research_note_path": "repo-relative markdown path for deterministic parsing."
  },
  "flagged_for_next_roadmap": "flagged_for_v421: combine Family-B executable world-model induction (arXiv:2605.05138) with bounded counterexample-guided refinement (arXiv:2606.11521) inside the Exp 4544 GOAL+DYNAMICS+plan proposer",
  "honest_verdict": "complete: sota_ingestion_llm_inducer_mapped",
  "inference_substrate": "aggregation_from_upstream_artifacts",
  "methods_mapped": [
    {
      "fails_when": "the induced executable model is trusted as proof without held-out transition checks, or when an L1 completion predicate survives a new ARC-AGI-3 goal-acquisition phase after the level shifts.",
      "method": "Family-B executable world-model induction for per-level re-induction",
      "source_ids": [
        "2605.05138",
        "2603.24621"
      ],
      "takes_over_current_a1_llm_proposer": "Exp 4544 already asks the LLM for separate GOAL and DYNAMICS candidates plus a plan. This method makes that slot stricter: the proposer emits an executable Python world model, verifies prior observations and held-out post-level-up transitions, refactors toward simpler state variables, then plans only through accepted GOAL+DYNAMICS candidates.",
      "v421_candidate": "flagged_for_v421: Family-B executable GOAL+DYNAMICS proposer inside the Exp 4544 re-induction hook"
    },
    {
      "fails_when": "the validation set is not separated from the evidence used to prompt the proposer, or when the proposal count grows into an unbounded best-of-N search that loses the action-efficiency target.",
      "method": "LLM-PV proposal distribution with held-out execution selection",
      "source_ids": [
        "2510.14331"
      ],
      "takes_over_current_a1_llm_proposer": "Exp 4544 uses a single local GGUF call per refinement round. LLM-PV reframes the call as a proposal distribution over compact candidate programs: sample bounded GOAL, DYNAMICS, and plan candidates, execute each on held-out transitions, and select the lowest verifier-energy candidate without gradient updates.",
      "v421_candidate": "flagged_for_v421: bounded LLM-PV sampling around each Exp 4544 GOAL+DYNAMICS refinement round"
    },
    {
      "fails_when": "the LLM-generated oracle is itself unchecked; in ARC-AGI-3 it must be validated against observed transitions before it can score plans or the loop becomes verifier self-confirmation.",
      "method": "ALGO-style LLM-generated oracle verifier for candidate plans",
      "source_ids": [
        "2305.14591"
      ],
      "takes_over_current_a1_llm_proposer": "Exp 4544 feeds verifier failures back to the proposer, but its candidate verifier is fixed. ALGO suggests asking the proposer for a small executable reference oracle or exhaustive checker alongside the plan, then using that checker to guide search and reject candidate algorithms before acting.",
      "v421_candidate": "flagged_for_v421: proposer-emitted oracle checkers validated by observed transitions before plan selection"
    },
    {
      "fails_when": "Prolog-style proof traces are copied without an ARC-AGI-3 state representation, or when refinement optimizes final-frame agreement instead of the GOAL and DYNAMICS clauses used by the live planner.",
      "method": "ABPR procedural refinement with executable trace feedback",
      "source_ids": [
        "2603.20334"
      ],
      "takes_over_current_a1_llm_proposer": "Exp 4544 has a bounded verifier-guided refinement loop, but the feedback can be only a first counterexample or reachability failure. ABPR upgrades that feedback into structured executable trace evidence about which subgoal, relation, or transformation failed, then asks the proposer to repair that specific part.",
      "v421_candidate": "flagged_for_v421: structured trace feedback for the existing three-round Exp 4544 refinement budget"
    },
    {
      "fails_when": "counterexamples are appended as extra prompt text without normalization, or when the teacher returns examples outside the observed ARC state/action language the proposer can execute.",
      "method": "counterexample-guided learning for verifier-returned failures",
      "source_ids": [
        "2606.11521"
      ],
      "takes_over_current_a1_llm_proposer": "Exp 4544 already returns a first verifier counterexample or plan-reachability failure. Counterexample Guided Learning turns that into the central teaching signal: minimize or cluster the failing trace, feed it back to the local proposer, and stop after the bounded refinement budget or a reachable accepted plan.",
      "v421_candidate": "flagged_for_v421: minimized verifier counterexamples as the primary repair signal for Exp 4544"
    }
  ],
  "preconditions_checked": {
    "a1_llm_proposer_artifact_read": true,
    "a1_llm_proposer_spec_read": true,
    "agents_md_read": true,
    "arxiv_api_reachable": true,
    "arxiv_http_200_verified_ids": [
      "2605.05138",
      "2510.14331",
      "2305.14591",
      "2603.20334",
      "2606.11521",
      "2603.24621"
    ],
    "codex_md_read": true,
    "deep_research_invoked": false,
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
      "http://export.arxiv.org/api/query?search_query=(abs:\"neural+guided+search\"+OR+abs:\"learned+heuristic\"+OR+abs:\"value+guided+search\"+OR+abs:\"program+induction\"+OR+abs:\"world+model\"+OR+abs:\"goal+induction\")+AND+(abs:\"planning\"+OR+abs:\"agent\"+OR+abs:\"reasoning\"+OR+abs:\"reinforcement+learning\")&start=0&max_results=8&sortBy=submittedDate&sortOrder=descending",
      "http://export.arxiv.org/api/query?search_query=(abs:\"active+inference\"+OR+abs:\"free+energy\"+OR+abs:\"free+energy+principle\"+OR+abs:\"predictive+coding\"+OR+abs:\"world+model\")+AND+(abs:\"LLM\"+OR+abs:\"language+model\"+OR+abs:\"reasoning\")&start=0&max_results=8&sortBy=submittedDate&sortOrder=descending"
    ],
    "sweep_clusters_used": true,
    "sweep_semscholar_arxiv_ids": [],
    "sweep_semscholar_queries": [
      "LLM world model induction ARC-AGI-3 executable world models",
      "LLM priors ERM over programs ALGO program synthesis",
      "procedural refinement counterexample guided learning verifier guided refinement",
      "goal acquisition ARC-AGI-3 goal shift detection"
    ],
    "sweep_semscholar_rate_limited_queries": [
      "LLM world model induction ARC-AGI-3 executable world models",
      "LLM priors ERM over programs ALGO program synthesis",
      "procedural refinement counterexample guided learning verifier guided refinement",
      "goal acquisition ARC-AGI-3 goal shift detection"
    ],
    "sweep_semscholar_used": true,
    "training_launched": false,
    "websearch_webfetch_top_sources": [
      "https://arxiv.org/abs/2605.05138",
      "https://arxiv.org/abs/2510.14331",
      "https://arxiv.org/abs/2305.14591",
      "https://arxiv.org/abs/2603.20334",
      "https://arxiv.org/abs/2606.11521",
      "https://arxiv.org/abs/2603.24621"
    ]
  },
  "random_seed": 4553,
  "research_note_path": "docs/research-notes/arc-llm-inducer-sota-420.md"
}
```

Reliable channel only: `scripts/sweep_clusters.py`, `scripts/sweep_semscholar.py`,
arXiv abs-page HTTP-200 checks, and low-concurrency WebSearch/WebFetch of six
top LLM-inducer / verifier-refinement / goal-acquisition sources. Preconditions
passed before any claim was promoted: `.venv/bin/python scripts/sweep_clusters.py --help`
exited zero and
`curl -sf -o /dev/null https://export.arxiv.org/api/query?search_query=all:test`
confirmed arXiv API reachability. `scripts/sweep_clusters.py 6 --max-results 8`
and `scripts/sweep_clusters.py 3 --max-results 8` emitted the focused world-model
cluster URLs. `scripts/sweep_semscholar.py` ran four focused queries and returned
HTTP 429 on all four, so no S2-only claim was promoted. No `/deep-research`
call was made. No training, live LLM inference, leaderboard submission, or live
solve was launched. No ops/status/traceability files or
`scripts/research_conductor.py` were modified.

Already-discovered corpus read through an LLM-world-model-induction /
verifier-guided-refinement / goal-shift filter: `research-studying.md`,
`research-references.md`, `openspec/capabilities/arc-world-model-trust-energy/spec.md`
at `REQ-ARC-WMTE-4544`, and
`results/experiment_4544_llm_proposer_reinduction.json`. The current mechanism
this maps onto is Exp 4544: A1 asks the local generator for separate
GOAL+DYNAMICS+plan candidates and runs a bounded verifier-guided refinement loop
where verifier counterexamples or plan-reachability failures are fed back to the
proposer for at most three rounds.

Sources checked: arXiv:2605.05138, arXiv:2510.14331, arXiv:2305.14591, arXiv:2603.20334, arXiv:2606.11521, arXiv:2603.24621.

## Per-Method Mapping

- **Family-B executable world-model induction** (arXiv:2605.05138,
  arXiv:2603.24621): take over the Exp 4544 post-level-up proposer with an
  executable Python world model that separates GOAL and DYNAMICS, verifies
  transitions, refactors toward simpler state, and plans only through accepted
  models. This is the strongest .421 candidate because it matches ARC-AGI-3's
  goal acquisition and world-model demands directly.
- **LLM-PV proposal distribution with held-out execution selection**
  (arXiv:2510.14331): treat the local LLM as a bounded proposal distribution
  over candidate programs, execute and score each candidate on held-out
  transitions, and select without gradient updates. It fails if the validation
  transitions leak into the prompt or if best-of-N becomes unbounded.
- **ALGO-style LLM-generated oracle verifier** (arXiv:2305.14591): have the
  proposer emit a small executable checker or reference oracle with the plan,
  then validate that checker against observations before using it to guide
  search. It fails if the LLM-generated oracle is accepted without independent
  transition checks.
- **ABPR procedural refinement with executable trace feedback**
  (arXiv:2603.20334): replace generic "plan failed" feedback with structured
  trace evidence about which subgoal, relation, or transformation failed. It
  fails if ARC-AGI-2 static-grid proof traces are copied without an ARC-AGI-3
  state/action representation.
- **Counterexample Guided Learning** (arXiv:2606.11521): make minimized verifier
  counterexamples the primary teaching signal inside the three-round Exp 4544
  bounded verifier-guided refinement loop. It fails when counterexamples are
  unnormalized prompt text rather than executable state/action evidence.

## .421 Candidate

flagged_for_v421: combine Family-B executable world-model induction (arXiv:2605.05138) with bounded counterexample-guided refinement (arXiv:2606.11521) inside the Exp 4544 GOAL+DYNAMICS+plan proposer

The practical next experiment should keep Exp 4544's trigger, local proposer,
oracle-distinct verifier energy, and three-round cap, but make the proposal unit
an executable GOAL+DYNAMICS world model plus a candidate plan. LLM-PV supplies
the bounded proposal-selection semantics, ALGO supplies an optional generated
checker only after independent validation, ABPR supplies structured trace
feedback, and Counterexample Guided Learning supplies the repair signal. The
goal-shift risk from ARC-AGI-3 means a stale L1 predicate must be treated as a
counterexample after any level-up.
