# Offline-live bridge literature ingestion 2026-06-23

```json
{
  "citations_verified": {
    "1011.0686": {
      "http_status": 200,
      "title": "A Reduction of Imitation Learning and Structured Prediction to No-Regret Online Learning",
      "url": "https://arxiv.org/abs/1011.0686"
    },
    "1706.04599": {
      "http_status": 200,
      "title": "On Calibration of Modern Neural Networks",
      "url": "https://arxiv.org/abs/1706.04599"
    },
    "2102.04518": {
      "http_status": 200,
      "title": "A* Search Without Expansions: Learning Heuristic Functions with Deep Q-Networks",
      "url": "https://arxiv.org/abs/2102.04518"
    },
    "2206.03023": {
      "http_status": 200,
      "title": "How Far I'll Go: Offline Goal-Conditioned Reinforcement Learning via f-Advantage Regression",
      "url": "https://arxiv.org/abs/2206.03023"
    },
    "2303.09477": {
      "http_status": 200,
      "title": "Learning Local Heuristics for Search-Based Navigation Planning",
      "url": "https://arxiv.org/abs/2303.09477"
    },
    "2406.04935": {
      "http_status": 200,
      "title": "SLOPE: Search with Learned Optimal Pruning-based Expansion",
      "url": "https://arxiv.org/abs/2406.04935"
    },
    "2511.10264": {
      "http_status": 200,
      "title": "Beyond Single-Step Updates: Reinforcement Learning of Heuristics with Limited-Horizon Search",
      "url": "https://arxiv.org/abs/2511.10264"
    },
    "2604.11351": {
      "http_status": 200,
      "title": "WM-DAgger: Enabling Efficient Data Aggregation for Imitation Learning with World Models",
      "url": "https://arxiv.org/abs/2604.11351"
    }
  },
  "deep_research_not_used": true,
  "field_principles": {
    "deep_research_not_used": {
      "principle": "MUST be true -- /deep-research is banned in the autonomous loop; used sweep helpers + low-concurrency WebSearch/WebFetch."
    },
    "flagged_for_next_roadmap": {
      "principle": "the strongest method(s) flagged as candidate .427 inputs -- closes discover->ingest->plan->experiment."
    },
    "honest_verdict": {
      "principle": "terminal prefix; success: sota_ingestion_offline_live_bridge_mapped."
    },
    "inference_substrate": {
      "principle": "aggregation_from_upstream_artifacts -- literature read + synthesis, no model load (100us floor)."
    },
    "methods_mapped": {
      "principle": "the strongest 3-5 SOTA methods with REAL arXiv IDs + per-method implement-cost-over-current-stack + fails_when -- the actionable ingestion (no citation = fabrication)."
    },
    "note_path": {
      "principle": "docs/research-notes/offline-live-bridge-literature-2026-06-23.md -- the per-track note (the SOTA-Ingestion Cycle deliverable)."
    },
    "preconditions_checked": {
      "principle": "records network reachability verified; pre-empts fabricated citations."
    }
  },
  "flagged_for_next_roadmap": [
    "flagged_for_v427: dagger_search_distribution_value_retraining (arXiv:1011.0686 + arXiv:2604.11351)",
    "flagged_for_v427: calibrated_value_to_cost_tiebreaker (arXiv:1706.04599)",
    "flagged_for_v427: decision_point_cached_qstar_value_head (arXiv:2102.04518 + arXiv:2511.10264)"
  ],
  "honest_verdict": "success: sota_ingestion_offline_live_bridge_mapped",
  "inference_substrate": "aggregation_from_upstream_artifacts",
  "methods_mapped": [
    {
      "fails_when": "expert or replay labels are unavailable, world-model-synthesized OOD recovery states are hallucinated, the aggregated frontier distribution keeps moving faster than retraining, or A2 cannot cache the expanded feature set cheaply.",
      "implement_cost_over_current_stack": "medium-high: instrument the A2 live search to log off-path frontier states, label corrective actions or costs with replay/expert evidence, aggregate those rows into the value-head training set, and optionally use A1-trusted world models to synthesize recovery states.",
      "maps_to_current_stack": "A1 disambiguation already names distribution_shift as one bridge cause; A2 graduated-value-head needs training data from the states its own live frontier visits, not only the winning-path traces used offline.",
      "method": "Search-distribution DAgger retraining for off-path frontier states",
      "roadmap_candidate": "flagged_for_v427: dagger_search_distribution_value_retraining (arXiv:1011.0686 + arXiv:2604.11351)",
      "source_ids": [
        "1011.0686",
        "2604.11351"
      ],
      "track": "distribution_shift"
    },
    {
      "fails_when": "the calibration set misses off-path frontier states, per-game score monotonicity is nonstationary, the calibrated cost overrides legality or depth controls, or the mapping improves ECE while leaving live first-win and action efficiency unchanged.",
      "implement_cost_over_current_stack": "low-to-medium: fit an isotonic, Platt-style, or temperature-scaling calibrator from value-head score to held-out steps-to-go or win probability, then clamp it into a bounded A2 tie-breaker cost.",
      "maps_to_current_stack": "A1 disambiguation separates calibration from representation quality; A2 can keep the graduated value head but stop treating an uncalibrated ranking score as an A* cost.",
      "method": "Post-hoc calibration from value ranking to bounded search cost",
      "roadmap_candidate": "flagged_for_v427: calibrated_value_to_cost_tiebreaker (arXiv:1706.04599)",
      "source_ids": [
        "1706.04599"
      ],
      "track": "calibration"
    },
    {
      "fails_when": "ARC action abstractions prevent batched scoring, cached features drift after hidden-state updates, the forward pass is still slower than bare search, or inadmissible values are promoted from tie-breakers to hard shortest-path claims.",
      "implement_cost_over_current_stack": "medium: batch or cache value evaluation at decision points, score candidate actions in one forward pass when possible, and refresh heuristic targets with limited-horizon search instead of per-node full feature recomputation.",
      "maps_to_current_stack": "A1 disambiguation points to compute_cost when the value head is slower than bare BFS; A2 can retain the graduated head only in a bounded, cached decision-point role rather than the regressed heavy A* mode.",
      "method": "Decision-point cached Q*/limited-horizon value evaluation",
      "roadmap_candidate": "flagged_for_v427: decision_point_cached_qstar_value_head (arXiv:2102.04518 + arXiv:2511.10264)",
      "source_ids": [
        "2102.04518",
        "2511.10264"
      ],
      "track": "compute_cost"
    },
    {
      "fails_when": "the pruner drops the only branch that exposes a hidden register, public training levels do not cover the live frontier geometry, or open-list memory improves without any live first-win or action-efficiency lift.",
      "implement_cost_over_current_stack": "medium-high: train a distance-from-promising-frontier or local heuristic model, use it only to shorten the open list after classical legality/depth filters, and gate adoption on matched bare-BFS and linear-baseline no-regression tests.",
      "maps_to_current_stack": "A1 identifies whether shift or compute binds; A2 can use learned pruning only after the graduated value head has a calibrated or search-distribution-aware signal.",
      "method": "SLOPE-style learned pruning behind no-regression gates",
      "roadmap_candidate": "candidate_for_v427_after_no_regression: slope_bounded_pruning",
      "source_ids": [
        "2406.04935",
        "2303.09477"
      ],
      "track": "bounded_pruning"
    },
    {
      "fails_when": "the GOAL predicate is wrong, hindsight or failure relabeling smears incompatible level goals together, the value ignores hidden registers, or dense goal value overrides the scored-agent preservation gate.",
      "implement_cost_over_current_stack": "medium: condition the SpatialValueNet-style head on the current A1 registered GOAL predicate or level target, train from offline traces and failure relabeling, and expose the result only as an A2 tie-breaker.",
      "maps_to_current_stack": "A1 supplies the live bridge diagnosis plus the goal/register predicate; A2 needs a dense value whose meaning changes when the level goal changes, instead of one global score for incompatible goals.",
      "method": "Goal-conditioned offline value tied to the induced GOAL predicate",
      "roadmap_candidate": "candidate_for_v427: goal_conditioned_spatial_value_tiebreaker",
      "source_ids": [
        "2206.03023"
      ],
      "track": "goal_conditioned_value"
    }
  ],
  "note_path": "docs/research-notes/offline-live-bridge-literature-2026-06-23.md",
  "preconditions_checked": {
    "agents_md_read": true,
    "arxiv_http_200_verified_ids": [
      "https://arxiv.org/abs/1011.0686",
      "https://arxiv.org/abs/2604.11351",
      "https://arxiv.org/abs/1706.04599",
      "https://arxiv.org/abs/2102.04518",
      "https://arxiv.org/abs/2406.04935",
      "https://arxiv.org/abs/2206.03023",
      "https://arxiv.org/abs/2511.10264",
      "https://arxiv.org/abs/2303.09477"
    ],
    "bridge_diagnosis_note_read": true,
    "codex_md_read": true,
    "deep_research_invoked": false,
    "exp4613_artifact_read": true,
    "leaderboard_submission": false,
    "live_llm_inference": false,
    "model_load": false,
    "network_hf_models_reachable": true,
    "ops_docs_modified": false,
    "research_conductor_modified": false,
    "research_references_read": true,
    "research_studying_read": true,
    "sweep_clusters_urls": [
      "http://export.arxiv.org/api/query?search_query=(abs:\"neural+guided+search\"+OR+abs:\"learned+heuristic\"+OR+abs:\"value+guided+search\"+OR+abs:\"program+induction\"+OR+abs:\"world+model\"+OR+abs:\"goal+induction\")+AND+(abs:\"planning\"+OR+abs:\"agent\"+OR+abs:\"reasoning\"+OR+abs:\"reinforcement+learning\")&start=0&max_results=8&sortBy=submittedDate&sortOrder=descending",
      "http://export.arxiv.org/api/query?search_query=(abs:\"affordance\"+OR+abs:\"action+effect\"+OR+abs:\"clickability\"+OR+abs:\"frame+prediction\"+OR+abs:\"intrinsic+motivation\"+OR+abs:\"directed+exploration\"+OR+abs:\"novelty+search\")+AND+(abs:\"reinforcement+learning\"+OR+abs:\"agent\"+OR+abs:\"exploration\"+OR+abs:\"interactive+environment\"+OR+abs:\"ARC\")&start=0&max_results=8&sortBy=submittedDate&sortOrder=descending"
    ],
    "sweep_clusters_used": true,
    "sweep_semscholar_arxiv_ids": [],
    "sweep_semscholar_queries": [
      "DAgger dataset aggregation imitation learning distribution shift 1011.0686",
      "DeepCubeA Q* learned heuristic A* 2102.04518 SLOPE 2406.04935",
      "GoFAR f-Advantage Regression goal-conditioned offline reinforcement learning 2206.03023",
      "post-hoc calibration neural networks Platt isotonic learned value cost search 1706.04599",
      "amortized learned heuristic search value guided search transfer 2026"
    ],
    "sweep_semscholar_rate_limited_queries": [
      "DAgger dataset aggregation imitation learning distribution shift 1011.0686",
      "DeepCubeA Q* learned heuristic A* 2102.04518 SLOPE 2406.04935",
      "GoFAR f-Advantage Regression goal-conditioned offline reinforcement learning 2206.03023",
      "post-hoc calibration neural networks Platt isotonic learned value cost search 1706.04599",
      "amortized learned heuristic search value guided search transfer 2026"
    ],
    "sweep_semscholar_used": true,
    "training_launched": false,
    "websearch_webfetch_top_sources": [
      "https://arxiv.org/abs/1011.0686",
      "https://arxiv.org/abs/2604.11351",
      "https://arxiv.org/abs/1706.04599",
      "https://arxiv.org/abs/2102.04518",
      "https://arxiv.org/abs/2406.04935",
      "https://arxiv.org/abs/2206.03023",
      "https://arxiv.org/abs/2511.10264",
      "https://arxiv.org/abs/2303.09477"
    ],
    "world_model_trust_note_read": true
  },
  "random_seed": 4625
}
```

## Fresh-pass provenance

Read `AGENTS.md`, `CODEX.md`, `results/experiment_4613_sota_ingestion_world_model_trust.json`,
`docs/research-notes/world-model-trust-literature-2026-06-23.md`,
`docs/research-notes/arc-representation-not-the-bottleneck-2026-06-23.md`,
`research-studying.md`, and `research-references.md`. The filtered track was
the .426 headline open problem: the offline-to-live bridge where a good offline
value/verifier regresses the live search through compute cost, distribution
shift, or calibration error, feeding candidate methods forward to .427.

Reliable-channel pass, not `/deep-research`:
- `.venv/bin/python -c "import urllib.request; urllib.request.urlopen('https://huggingface.co/api/models', timeout=10); print('net_ok')"`
- `.venv/bin/python scripts/sweep_clusters.py 6 --max-results 8`
- `.venv/bin/python scripts/sweep_clusters.py 5 --max-results 8`
- `.venv/bin/python scripts/sweep_semscholar.py - --limit 8` with five focused queries
- low-concurrency WebSearch/WebFetch of the top arXiv papers
- direct arXiv HTTP checks for all cited IDs

Semantic Scholar returned HTTP 500 or 429 for the five focused queries, so no
Semantic-Scholar-only source was promoted. Direct arXiv HTTP checks returned
200 for arXiv:1011.0686, arXiv:2604.11351, arXiv:1706.04599,
arXiv:2102.04518, arXiv:2406.04935, arXiv:2206.03023, arXiv:2511.10264, and
arXiv:2303.09477. No live LLM inference, No training, No leaderboard submission,
no model load, and no live solve claim were run or made. `scripts/research_conductor.py`,
`ops/changelog.md`, and `ops/status.md` were not edited by this workflow.

## SOTA -> experiment mapping

## DAgger and WM-DAgger search-distribution retraining

**Sources:** DAgger, arXiv:1011.0686; WM-DAgger, arXiv:2604.11351.

**Mapping to A1 disambiguation / A2 graduated-value-head:** A1 already names
distribution shift as a candidate bridge cause: the value head was trained on
winning-path states but the live frontier spends most time off that manifold.
DAgger says to train on the distribution induced by the learned policy; the
A2 analogue is to log live frontier states, label them with replay/expert or
trusted-model corrective evidence, and aggregate them into the SpatialValueNet
training set. WM-DAgger adds the 2026 variant: use a world model to synthesize
OOD recovery rows, but only with consistency filtering.

**Implementation cost over current stack:** medium-high. Needs live frontier
logging, corrective labels or costs, retraining, and a cache-aware A2 path.

**Fails when:** synthesized recovery states are not execution-consistent,
expert labels are unavailable, or retraining chases a shifting live frontier.

## Post-hoc value-to-cost calibration

**Source:** neural post-hoc calibration, arXiv:1706.04599. The classic
Platt/isotonic names are older than arXiv-native coverage; the arXiv-backed
claim here is post-hoc calibration of neural scores, with temperature scaling
as the Platt-style single-parameter variant.

**Mapping to A1 disambiguation / A2 graduated-value-head:** A1's calibration
arm distinguishes a useful ranker from a usable search cost. A2 should fit a
held-out monotone mapping from value score to steps-to-go or win probability,
then clamp the result into a bounded tie-breaker rather than a heavy A* priority.

**Implementation cost over current stack:** low-to-medium. Reuse cached A1/A2
traces, fit leave-game-out calibration, and wire the calibrated output only
where the live path already supports bounded value use.

**Fails when:** off-path states are absent from the calibration set, per-game
monotonicity flips, or the calibrated cost is allowed to override legality and
depth controls.

## Cached Q*/limited-horizon value evaluation

**Sources:** Q*/DeepCubeA-style learned heuristic search, arXiv:2102.04518;
limited-horizon heuristic updates, arXiv:2511.10264.

**Mapping to A1 disambiguation / A2 graduated-value-head:** A1's compute-cost
arm says the value can regress live search by consuming the time budget. Q*
pushes toward amortized action scoring; limited-horizon updates push toward
training targets that reflect real search fronts. A2 should evaluate the value
head only at decision points, cache feature vectors, and batch candidate scoring
where the action set permits it.

**Implementation cost over current stack:** medium. Requires cache keys for
`cross_game_features_v3`, batched scoring, and regression tests against bare BFS.

**Fails when:** hidden state invalidates cached features, action abstraction
prevents batching, or an inadmissible value is promoted from tie-breaker to
shortest-path proof.

## SLOPE/local-heuristic bounded pruning

**Sources:** SLOPE learned optimal-pruning expansion, arXiv:2406.04935; local
heuristics for generalizing search-based planning, arXiv:2303.09477.

**Mapping to A1 disambiguation / A2 graduated-value-head:** SLOPE attacks the
same compute surface as A2: the open list and child expansion budget. It should
come after DAgger/calibration because pruning is more dangerous than ranking.
The safe implementation is shortlist-only pruning behind hard no-regression
controls, never pruning before legality/depth gates run.

**Implementation cost over current stack:** medium-high. Needs labels for
near-good-frontier distance, matched bare controls, and hidden-register branch
retention checks.

**Fails when:** pruning removes the branch that reveals a hidden register or
only improves memory while first-win and action efficiency stay flat.

## Goal-conditioned offline value at level boundaries

**Source:** GoFAR goal-conditioned offline value, arXiv:2206.03023.

**Mapping to A1 disambiguation / A2 graduated-value-head:** A1 supplies the
currently induced GOAL predicate or register-aware level target. A2 needs a
dense value whose meaning changes when that target changes; otherwise a value
trained for L1 can steer away from L2. GoFAR supports offline goal-conditioned
value learning without pretending a single global score generalizes across
incompatible goals.

**Implementation cost over current stack:** medium. Add goal encoding to the
SpatialValueNet input, train from offline traces and failure relabeling, then
expose only as a bounded tie-breaker until per-level no-regression passes.

**Fails when:** the GOAL predicate is wrong, relabeling merges incompatible
level goals, or dense value overrides the scored-agent preservation gate.

## Bottom line for the .427 roadmap

1. Build `flagged_for_v427: dagger_search_distribution_value_retraining`
   first if A1 confirms distribution shift: DAgger arXiv:1011.0686 plus
   WM-DAgger arXiv:2604.11351 gives the search-distribution data recipe.
2. Build `flagged_for_v427: calibrated_value_to_cost_tiebreaker` first if A1
   confirms calibration: arXiv:1706.04599 supports post-hoc calibration of the
   neural value into a bounded cost-like signal.
3. Build `flagged_for_v427: decision_point_cached_qstar_value_head` first if A1
   confirms compute cost: Q*/DeepCubeA arXiv:2102.04518 and limited-horizon
   heuristic learning arXiv:2511.10264 are the compute-aware value route.
4. Keep SLOPE arXiv:2406.04935 plus local heuristics arXiv:2303.09477 as the
   second-stage pruning lever after no-regression evidence exists.
5. Fold GoFAR arXiv:2206.03023 into the .427 value roadmap when the bridge fix
   needs goal-conditioned behavior across level boundaries.

