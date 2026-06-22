# Feature-router and env-adaptive replay SOTA ingestion .423 - 2026-06-22

```json
{
  "citations_verified": {
    "2512.24156": {
      "http_status": 200,
      "title": "Graph-Based Exploration for ARC-AGI-3 Interactive Reasoning Tasks",
      "url": "https://arxiv.org/abs/2512.24156"
    },
    "2602.01869": {
      "http_status": 200,
      "title": "Skill-Pro: Learning Reusable Skills from Experience via Non-Parametric PPO for LLM Agents",
      "url": "https://arxiv.org/abs/2602.01869"
    },
    "2602.08234": {
      "http_status": 200,
      "title": "SkillRL: Evolving Agents via Recursive Skill-Augmented Reinforcement Learning",
      "url": "https://arxiv.org/abs/2602.08234"
    },
    "2603.22455": {
      "http_status": 200,
      "title": "SkillRouter: Skill Routing for LLM Agents at Scale",
      "url": "https://arxiv.org/abs/2603.22455"
    },
    "2603.24621": {
      "http_status": 200,
      "title": "ARC-AGI-3: A New Challenge for Frontier Agentic Intelligence",
      "url": "https://arxiv.org/abs/2603.24621"
    },
    "2605.12039": {
      "http_status": 200,
      "title": "SkillGraph: Skill-Augmented Reinforcement Learning for Agents via Evolving Skill Graphs",
      "url": "https://arxiv.org/abs/2605.12039"
    },
    "2606.06079": {
      "http_status": 200,
      "title": "SkillComposer: Learning to Evolve Agent Skills for Specification and Generalization",
      "url": "https://arxiv.org/abs/2606.06079"
    }
  },
  "field_principles": {
    "citations_verified": "every method claim cites a verifiable arXiv ID/URL -- the no-fabrication bar (same as any results artifact).",
    "field_principles": "principle annotations for every top-level artifact field.",
    "flagged_for_next_roadmap": "the strongest method flagged as a .424 candidate -- closes the discover->ingest->plan loop.",
    "honest_verdict": "terminal prefix; complete: sota_ingestion_feature_router_mapped.",
    "inference_substrate": "aggregation_from_upstream_artifacts -- literature synthesis, no model load (100us floor).",
    "methods_mapped": "the 3-5 strongest methods with REAL arXiv IDs -- the shoulders-of-giants anti-rederivation check.",
    "preconditions_checked": "records resources verified; pre-empts missing-resource fabrication.",
    "random_seed": "bare integer seed for reproducible artifact generation.",
    "research_note_path": "repo-relative markdown path for deterministic parsing."
  },
  "flagged_for_next_roadmap": "flagged_for_v424: implement SkillRouter-style full-body routing over arc_solver_kit skills, backed by SkillGraph/SkillRL trace distillation and graph-explore env-adaptive replay regeneration for drifted rows (arXiv:2603.22455 + arXiv:2605.12039 + arXiv:2512.24156)",
  "honest_verdict": "complete: sota_ingestion_feature_router_mapped",
  "inference_substrate": "aggregation_from_upstream_artifacts",
  "methods_mapped": [
    {
      "fails_when": "skill bodies are hidden at routing time, the library is not execution validated, retrieval is evaluated only on seen public games, or routing is left as a final candidate reranker after the winning action is absent.",
      "method": "SkillRouter full-text retrieve-and-rerank over the solver toolkit",
      "source_ids": [
        "2603.22455"
      ],
      "takes_over_current_a1_a3_mechanisms": "Exp 4582 currently classifies first-K early-play effects into a small mechanic class and routes to a fixed approach, but the artifact ended as a null because winner generation stayed mostly absent. SkillRouter takes over Exp 4582 by retrieving and reranking full skill bodies from the arc_solver_kit toolkit before planning, rather than exposing only short names or hand-coded class labels.",
      "target_track": "feature_skill_routing",
      "v424_candidate": "flagged_for_v424: SkillRouter-style full-body routing over arc_solver_kit skills for the Exp 4582 seen-to-hidden feature router"
    },
    {
      "fails_when": "the graph grows from raw trajectories without deduplication, failed traces are not distilled into negative routing constraints, or dependency edges are trusted without replay through the current ARC environment.",
      "method": "SkillGraph plus SkillRL evolving skill-library structure",
      "source_ids": [
        "2605.12039",
        "2602.08234"
      ],
      "takes_over_current_a1_a3_mechanisms": "Exp 4582 stores a flat mechanic-class to approach preference learned from positive and negative traces. SkillGraph and SkillRL take over that Exp 4582 policy by turning traces into a structured skill graph or SkillBank with dependency edges, failure lessons, and adaptive retrieval for general and task-specific heuristics.",
      "target_track": "feature_skill_routing",
      "v424_candidate": "flagged_for_v424: SkillGraph/SkillRL library maintenance behind the Exp 4582 router, with replay-gated skill insertion"
    },
    {
      "fails_when": "merged skills become too abstract to trigger reliably, task-specific skills are allowed to leak public-game identity, or reusable procedures lack an offline replay gate before entering the toolkit.",
      "method": "SkillComposer and Skill-Pro skill merge into executable reusable procedures",
      "source_ids": [
        "2606.06079",
        "2602.01869"
      ],
      "takes_over_current_a1_a3_mechanisms": "Exp 4582 exposes residual gaps by mechanic class, and Exp 4580 banks trajectory evidence when a procedure is replayable. SkillComposer and Skill-Pro take over this handoff by creating, improving, merging, and verifying reusable skills with activation/execution/termination conditions instead of repeatedly deriving per-game recipes.",
      "target_track": "feature_skill_routing",
      "v424_candidate": "flagged_for_v424: SkillComposer/Skill-Pro merge pass over Exp 4582 mechanic gaps before persisting new solver primitives"
    },
    {
      "fails_when": "the game requires hidden carry-state induction, graph keys ignore layout version drift, exploration budgets do not preserve action efficiency, or newly recovered paths are not rechecked through offline reproduction.",
      "method": "Graph-Based Exploration as env-derived robust replay generator",
      "source_ids": [
        "2512.24156"
      ],
      "takes_over_current_a1_a3_mechanisms": "Exp 4580 succeeded by closing the package gap and recovering sc25 with an env-adaptive replay path instead of trusting stale flat coordinates. Graph-Based Exploration takes over Exp 4580's replay fragility by re-deriving state-action paths from current frames, visited-state graphs, and untested action priorities when frozen replay no longer matches.",
      "target_track": "env_adaptive_replay",
      "v424_candidate": "flagged_for_v424: graph-explore replay regeneration for Exp 4580 version-drift rows before falling back to stale coordinate banks"
    },
    {
      "fails_when": "the next roadmap optimizes only the public-game package gap, ignores actions-to-first-levelup, treats official and community harness evidence as identical, or promotes env-adaptive replay without held-out layout drift.",
      "method": "ARC-AGI-3 efficiency-and-drift evaluation contract",
      "source_ids": [
        "2603.24621"
      ],
      "takes_over_current_a1_a3_mechanisms": "Exp 4580's headline is not a new solve; it converts 53 reproduced public levels into 53 live-submittable levels by requiring environment-matched trajectories or an env-adaptive resolver. The ARC-AGI-3 report takes over the Exp 4580 acceptance contract: novel interactive environments require goal inference, dynamics modeling, planning, and action-efficient replay under changing layouts.",
      "target_track": "env_adaptive_replay",
      "v424_candidate": "flagged_for_v424: ARC-AGI-3 drift-aware live-submittable gate for every Exp 4580 replay primitive"
    }
  ],
  "preconditions_checked": {
    "agents_md_read": true,
    "arxiv_api_reachable": true,
    "arxiv_http_200_verified_ids": [
      "2603.22455",
      "2605.12039",
      "2606.06079",
      "2602.01869",
      "2602.08234",
      "2512.24156",
      "2603.24621"
    ],
    "codex_md_read": true,
    "deep_research_invoked": false,
    "exp4580_artifact_read": true,
    "exp4580_spec_read": true,
    "exp4582_artifact_read": true,
    "exp4582_spec_read": true,
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
      "SkillRouter Skill Routing LLM Agents at Scale",
      "SkillGraph skill augmented RL evolving skill graphs",
      "SkillComposer learning to evolve agent skills specification generalization Skill-Pro reusable skills non-parametric PPO",
      "ARC-AGI-3 graph-based exploration environment drift replay skill routing"
    ],
    "sweep_semscholar_rate_limited_queries": [
      "SkillRouter Skill Routing LLM Agents at Scale",
      "SkillGraph skill augmented RL evolving skill graphs",
      "SkillComposer learning to evolve agent skills specification generalization Skill-Pro reusable skills non-parametric PPO"
    ],
    "sweep_semscholar_used": true,
    "training_launched": false,
    "websearch_webfetch_top_sources": [
      "https://arxiv.org/abs/2603.22455",
      "https://arxiv.org/abs/2605.12039",
      "https://arxiv.org/abs/2606.06079",
      "https://arxiv.org/abs/2602.01869",
      "https://arxiv.org/abs/2602.08234",
      "https://arxiv.org/abs/2512.24156",
      "https://arxiv.org/abs/2603.24621"
    ]
  },
  "random_seed": 4589,
  "research_note_path": "docs/research-notes/sota-ingestion-feature-router-423-2026-06-22.md"
}
```

Reliable channel only: `scripts/sweep_clusters.py`, `scripts/sweep_semscholar.py`,
arXiv abs-page HTTP-200 checks, and low-concurrency WebSearch/WebFetch of seven
top skill-routing / skill-library / env-adaptive replay sources. Preconditions
passed before any claim was promoted: `.venv/bin/python scripts/sweep_clusters.py
--help` exited zero and `curl -sf -o /dev/null
https://export.arxiv.org/api/query?search_query=all:test` confirmed arXiv API
reachability. `scripts/sweep_clusters.py 5 --max-results 8` and
`scripts/sweep_clusters.py 6 --max-results 8` emitted the focused cluster URLs.
`scripts/sweep_semscholar.py` ran four focused queries; three returned HTTP 429
and no S2-only arXiv ID was promoted. No `/deep-research` call was made. No training,
live LLM inference, leaderboard submission, or live solve was launched.
No ops/status/traceability files or `scripts/research_conductor.py` were modified.

Already-discovered corpus read through a skill-routing and env-adaptive replay
filter: `research-studying.md`, `research-references.md`,
`openspec/capabilities/capstone/spec.md` at `REQ-CAPSTONE-4580` and
`REQ-CAPSTONE-4582`, `results/experiment_4580_live_submission_gap_close.json`,
and `results/experiment_4582_feature_router_transfer.json`. Exp 4580 succeeded
as a packaging/replay result: live-submittable levels rose from 33 to 53 and
`sc25` was recovered by env-adaptive replay. Exp 4582 was an honest no-value
feature-router null: router and baseline both measured 0.04 generic transfer,
random-route control did not pass, false-negative risk stayed open, and residual
generation gaps remain by mechanic class.

Sources checked: arXiv:2603.22455, arXiv:2605.12039, arXiv:2606.06079, arXiv:2602.01869, arXiv:2602.08234, arXiv:2512.24156, arXiv:2603.24621.

## Per-Method Mapping

- **SkillRouter full-text retrieve-and-rerank** (arXiv:2603.22455): the strongest
  A3 lesson is that Exp 4582 should route over full solver-skill bodies, not only
  a small mechanic-class label. It fails if hidden skill bodies or stale metadata
  become the routing substrate, or if routing happens after the winning action is
  already absent from the candidate pool.
- **SkillGraph plus SkillRL evolving skill-library structure** (arXiv:2605.12039,
  arXiv:2602.08234): turn Exp 4582 positive and negative traces into a graph or
  SkillBank with dependency edges, failure lessons, and adaptive retrieval. It
  fails if graph edges are trusted without current-environment replay validation.
- **SkillComposer and Skill-Pro executable skill reuse** (arXiv:2606.06079,
  arXiv:2602.01869): create, improve, merge, and verify skills so recurring ARC
  mechanics become reusable procedures with activation/execution/termination
  conditions. It fails when merged skills are too abstract or public-game identity
  leaks into the reusable primitive.
- **Graph-Based Exploration as robust replay regeneration** (arXiv:2512.24156):
  take Exp 4580's env-adaptive replay success and re-derive action paths from the
  current frame/state graph when flat coordinates drift. It fails on hidden-state
  or mechanic-limited games unless the graph key and action probes expose the
  latent state.
- **ARC-AGI-3 efficiency-and-drift evaluation contract** (arXiv:2603.24621): keep
  Exp 4580 honest by treating live-submittable replay, environment match, and
  actions-to-first-levelup as the score-facing contract. It fails if .424 optimizes
  only the public-game package gap and ignores hidden-layout or action-efficiency
  generalization.

## .424 Candidate

flagged_for_v424: implement SkillRouter-style full-body routing over arc_solver_kit skills, backed by SkillGraph/SkillRL trace distillation and graph-explore env-adaptive replay regeneration for drifted rows (arXiv:2603.22455 + arXiv:2605.12039 + arXiv:2512.24156)

The practical next experiment should keep Exp 4580's offline reproduction and
env-match gates, keep Exp 4582's random-route and false-negative controls, and
replace the flat route table with full-body skill retrieval plus graph-validated
replay regeneration. SkillRouter supplies the selection mechanism, SkillGraph and
SkillRL supply the evolving library structure, SkillComposer and Skill-Pro supply
the skill create/merge/reuse operations, Graph-Based Exploration supplies the
layout-drift replay fallback, and ARC-AGI-3 supplies the action-efficient
evaluation contract.
