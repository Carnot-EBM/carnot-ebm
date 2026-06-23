"""Exp 4625 offline-to-live bridge SOTA ingestion.

Spec refs: REQ-ARC-WMTE-4625, SCENARIO-ARC-WMTE-4625.

This is a literature-to-experiment mapping artifact, not a benchmark run. It
records the 2026-06-23 focused pass over distribution-shift, calibration, and
compute-cost fixes for the ARC offline-to-live bridge, then maps them onto the
current A1 disambiguation and A2 graduated-value-head stack for .427 planning.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import json
import os
from pathlib import Path
from typing import Any


JsonMap = Mapping[str, Any]

RESULT_RELATIVE_PATH = "results/experiment_4625_sota_ingestion_offline_live_bridge.json"
NOTE_RELATIVE_PATH = "docs/research-notes/offline-live-bridge-literature-2026-06-23.md"
RANDOM_SEED = 4625
HONEST_VERDICT = "success: sota_ingestion_offline_live_bridge_mapped"
INFERENCE_SUBSTRATE = "aggregation_from_upstream_artifacts"
TERMINAL_PREFIXES = (
    "complete:",
    "complete_",
    "success:",
    "success_",
    "passed:",
    "passed_",
    "shipped:",
    "shipped_",
    "blocked_",
)
REQUIRED_PRINCIPLE_FIELDS = frozenset(
    {
        "honest_verdict",
        "inference_substrate",
        "methods_mapped",
        "flagged_for_next_roadmap",
        "note_path",
        "deep_research_not_used",
        "preconditions_checked",
    }
)
REQUIRED_ARTIFACT_FIELDS = (
    "honest_verdict",
    "inference_substrate",
    "methods_mapped",
    "flagged_for_next_roadmap",
    "note_path",
    "deep_research_not_used",
    "preconditions_checked",
    "citations_verified",
    "random_seed",
    "field_principles",
)
REQUIRED_METHOD_FIELDS = frozenset(
    {
        "method",
        "source_ids",
        "track",
        "implement_cost_over_current_stack",
        "maps_to_current_stack",
        "fails_when",
        "roadmap_candidate",
    }
)
REQUIRED_CITATION_FIELDS = frozenset({"title", "url", "http_status"})
REQUIRED_PRECONDITION_FIELDS = frozenset(
    {
        "agents_md_read",
        "codex_md_read",
        "network_hf_models_reachable",
        "exp4613_artifact_read",
        "world_model_trust_note_read",
        "bridge_diagnosis_note_read",
        "research_studying_read",
        "research_references_read",
        "sweep_clusters_used",
        "sweep_clusters_urls",
        "sweep_semscholar_used",
        "sweep_semscholar_queries",
        "sweep_semscholar_arxiv_ids",
        "sweep_semscholar_rate_limited_queries",
        "arxiv_http_200_verified_ids",
        "websearch_webfetch_top_sources",
        "deep_research_invoked",
        "live_llm_inference",
        "training_launched",
        "model_load",
        "leaderboard_submission",
        "ops_docs_modified",
        "research_conductor_modified",
    }
)
REQUIRED_SOURCE_IDS = frozenset(
    {
        "1011.0686",
        "2604.11351",
        "1706.04599",
        "2102.04518",
        "2406.04935",
        "2206.03023",
        "2511.10264",
        "2303.09477",
    }
)
NOTE_REQUIRED_SOURCE_CITATIONS = frozenset(
    f"arXiv:{source_id}" for source_id in REQUIRED_SOURCE_IDS
)
FLAGGED_FOR_NEXT_ROADMAP = [
    (
        "flagged_for_v427: dagger_search_distribution_value_retraining "
        "(arXiv:1011.0686 + arXiv:2604.11351)"
    ),
    "flagged_for_v427: calibrated_value_to_cost_tiebreaker (arXiv:1706.04599)",
    (
        "flagged_for_v427: decision_point_cached_qstar_value_head "
        "(arXiv:2102.04518 + arXiv:2511.10264)"
    ),
]

FIELD_PRINCIPLES = {
    "honest_verdict": {
        "principle": "terminal prefix; success: sota_ingestion_offline_live_bridge_mapped."
    },
    "inference_substrate": {
        "principle": (
            "aggregation_from_upstream_artifacts -- literature read + synthesis, "
            "no model load (100us floor)."
        )
    },
    "methods_mapped": {
        "principle": (
            "the strongest 3-5 SOTA methods with REAL arXiv IDs + per-method "
            "implement-cost-over-current-stack + fails_when -- the actionable "
            "ingestion (no citation = fabrication)."
        )
    },
    "flagged_for_next_roadmap": {
        "principle": (
            "the strongest method(s) flagged as candidate .427 inputs -- closes "
            "discover->ingest->plan->experiment."
        )
    },
    "note_path": {
        "principle": (
            "docs/research-notes/offline-live-bridge-literature-2026-06-23.md -- "
            "the per-track note (the SOTA-Ingestion Cycle deliverable)."
        )
    },
    "deep_research_not_used": {
        "principle": (
            "MUST be true -- /deep-research is banned in the autonomous loop; used "
            "sweep helpers + low-concurrency WebSearch/WebFetch."
        )
    },
    "preconditions_checked": {
        "principle": "records network reachability verified; pre-empts fabricated citations."
    },
}

CITATIONS_VERIFIED = {
    "1011.0686": {
        "title": "A Reduction of Imitation Learning and Structured Prediction to No-Regret Online Learning",
        "url": "https://arxiv.org/abs/1011.0686",
        "http_status": 200,
    },
    "2604.11351": {
        "title": "WM-DAgger: Enabling Efficient Data Aggregation for Imitation Learning with World Models",
        "url": "https://arxiv.org/abs/2604.11351",
        "http_status": 200,
    },
    "1706.04599": {
        "title": "On Calibration of Modern Neural Networks",
        "url": "https://arxiv.org/abs/1706.04599",
        "http_status": 200,
    },
    "2102.04518": {
        "title": "A* Search Without Expansions: Learning Heuristic Functions with Deep Q-Networks",
        "url": "https://arxiv.org/abs/2102.04518",
        "http_status": 200,
    },
    "2406.04935": {
        "title": "SLOPE: Search with Learned Optimal Pruning-based Expansion",
        "url": "https://arxiv.org/abs/2406.04935",
        "http_status": 200,
    },
    "2206.03023": {
        "title": "How Far I'll Go: Offline Goal-Conditioned Reinforcement Learning via f-Advantage Regression",
        "url": "https://arxiv.org/abs/2206.03023",
        "http_status": 200,
    },
    "2511.10264": {
        "title": "Beyond Single-Step Updates: Reinforcement Learning of Heuristics with Limited-Horizon Search",
        "url": "https://arxiv.org/abs/2511.10264",
        "http_status": 200,
    },
    "2303.09477": {
        "title": "Learning Local Heuristics for Search-Based Navigation Planning",
        "url": "https://arxiv.org/abs/2303.09477",
        "http_status": 200,
    },
}

SWEEP_CLUSTER_URLS = [
    (
        "http://export.arxiv.org/api/query?search_query="
        '(abs:"neural+guided+search"+OR+abs:"learned+heuristic"+OR+'
        'abs:"value+guided+search"+OR+abs:"program+induction"+OR+'
        'abs:"world+model"+OR+abs:"goal+induction")+AND+'
        '(abs:"planning"+OR+abs:"agent"+OR+abs:"reasoning"+OR+'
        'abs:"reinforcement+learning")&start=0&max_results=8'
        "&sortBy=submittedDate&sortOrder=descending"
    ),
    (
        "http://export.arxiv.org/api/query?search_query="
        '(abs:"affordance"+OR+abs:"action+effect"+OR+abs:"clickability"+OR+'
        'abs:"frame+prediction"+OR+abs:"intrinsic+motivation"+OR+'
        'abs:"directed+exploration"+OR+abs:"novelty+search")+AND+'
        '(abs:"reinforcement+learning"+OR+abs:"agent"+OR+abs:"exploration"+OR+'
        'abs:"interactive+environment"+OR+abs:"ARC")&start=0&max_results=8'
        "&sortBy=submittedDate&sortOrder=descending"
    ),
]
S2_QUERIES = [
    "DAgger dataset aggregation imitation learning distribution shift 1011.0686",
    "DeepCubeA Q* learned heuristic A* 2102.04518 SLOPE 2406.04935",
    "GoFAR f-Advantage Regression goal-conditioned offline reinforcement learning 2206.03023",
    "post-hoc calibration neural networks Platt isotonic learned value cost search 1706.04599",
    "amortized learned heuristic search value guided search transfer 2026",
]
WEBSEARCH_WEBFETCH_TOP_SOURCES = [
    f"https://arxiv.org/abs/{source_id}"
    for source_id in (
        "1011.0686",
        "2604.11351",
        "1706.04599",
        "2102.04518",
        "2406.04935",
        "2206.03023",
        "2511.10264",
        "2303.09477",
    )
]

DEFAULT_PRECONDITIONS_CHECKED = {
    "agents_md_read": True,
    "codex_md_read": True,
    "network_hf_models_reachable": True,
    "exp4613_artifact_read": True,
    "world_model_trust_note_read": True,
    "bridge_diagnosis_note_read": True,
    "research_studying_read": True,
    "research_references_read": True,
    "sweep_clusters_used": True,
    "sweep_clusters_urls": SWEEP_CLUSTER_URLS,
    "sweep_semscholar_used": True,
    "sweep_semscholar_queries": S2_QUERIES,
    "sweep_semscholar_arxiv_ids": [],
    "sweep_semscholar_rate_limited_queries": S2_QUERIES,
    "arxiv_http_200_verified_ids": list(WEBSEARCH_WEBFETCH_TOP_SOURCES),
    "websearch_webfetch_top_sources": WEBSEARCH_WEBFETCH_TOP_SOURCES,
    "deep_research_invoked": False,
    "live_llm_inference": False,
    "training_launched": False,
    "model_load": False,
    "leaderboard_submission": False,
    "ops_docs_modified": False,
    "research_conductor_modified": False,
}

DEFAULT_METHODS_MAPPED = [
    {
        "method": "Search-distribution DAgger retraining for off-path frontier states",
        "source_ids": ["1011.0686", "2604.11351"],
        "track": "distribution_shift",
        "implement_cost_over_current_stack": (
            "medium-high: instrument the A2 live search to log off-path frontier "
            "states, label corrective actions or costs with replay/expert evidence, "
            "aggregate those rows into the value-head training set, and optionally "
            "use A1-trusted world models to synthesize recovery states."
        ),
        "maps_to_current_stack": (
            "A1 disambiguation already names distribution_shift as one bridge cause; "
            "A2 graduated-value-head needs training data from the states its own "
            "live frontier visits, not only the winning-path traces used offline."
        ),
        "fails_when": (
            "expert or replay labels are unavailable, world-model-synthesized OOD "
            "recovery states are hallucinated, the aggregated frontier distribution "
            "keeps moving faster than retraining, or A2 cannot cache the expanded "
            "feature set cheaply."
        ),
        "roadmap_candidate": FLAGGED_FOR_NEXT_ROADMAP[0],
    },
    {
        "method": "Post-hoc calibration from value ranking to bounded search cost",
        "source_ids": ["1706.04599"],
        "track": "calibration",
        "implement_cost_over_current_stack": (
            "low-to-medium: fit an isotonic, Platt-style, or temperature-scaling "
            "calibrator from value-head score to held-out steps-to-go or win "
            "probability, then clamp it into a bounded A2 tie-breaker cost."
        ),
        "maps_to_current_stack": (
            "A1 disambiguation separates calibration from representation quality; "
            "A2 can keep the graduated value head but stop treating an uncalibrated "
            "ranking score as an A* cost."
        ),
        "fails_when": (
            "the calibration set misses off-path frontier states, per-game score "
            "monotonicity is nonstationary, the calibrated cost overrides legality "
            "or depth controls, or the mapping improves ECE while leaving live "
            "first-win and action efficiency unchanged."
        ),
        "roadmap_candidate": FLAGGED_FOR_NEXT_ROADMAP[1],
    },
    {
        "method": "Decision-point cached Q*/limited-horizon value evaluation",
        "source_ids": ["2102.04518", "2511.10264"],
        "track": "compute_cost",
        "implement_cost_over_current_stack": (
            "medium: batch or cache value evaluation at decision points, score "
            "candidate actions in one forward pass when possible, and refresh "
            "heuristic targets with limited-horizon search instead of per-node "
            "full feature recomputation."
        ),
        "maps_to_current_stack": (
            "A1 disambiguation points to compute_cost when the value head is slower "
            "than bare BFS; A2 can retain the graduated head only in a bounded, "
            "cached decision-point role rather than the regressed heavy A* mode."
        ),
        "fails_when": (
            "ARC action abstractions prevent batched scoring, cached features drift "
            "after hidden-state updates, the forward pass is still slower than "
            "bare search, or inadmissible values are promoted from tie-breakers to "
            "hard shortest-path claims."
        ),
        "roadmap_candidate": FLAGGED_FOR_NEXT_ROADMAP[2],
    },
    {
        "method": "SLOPE-style learned pruning behind no-regression gates",
        "source_ids": ["2406.04935", "2303.09477"],
        "track": "bounded_pruning",
        "implement_cost_over_current_stack": (
            "medium-high: train a distance-from-promising-frontier or local "
            "heuristic model, use it only to shorten the open list after classical "
            "legality/depth filters, and gate adoption on matched bare-BFS and "
            "linear-baseline no-regression tests."
        ),
        "maps_to_current_stack": (
            "A1 identifies whether shift or compute binds; A2 can use learned "
            "pruning only after the graduated value head has a calibrated or "
            "search-distribution-aware signal."
        ),
        "fails_when": (
            "the pruner drops the only branch that exposes a hidden register, public "
            "training levels do not cover the live frontier geometry, or open-list "
            "memory improves without any live first-win or action-efficiency lift."
        ),
        "roadmap_candidate": "candidate_for_v427_after_no_regression: slope_bounded_pruning",
    },
    {
        "method": "Goal-conditioned offline value tied to the induced GOAL predicate",
        "source_ids": ["2206.03023"],
        "track": "goal_conditioned_value",
        "implement_cost_over_current_stack": (
            "medium: condition the SpatialValueNet-style head on the current A1 "
            "registered GOAL predicate or level target, train from offline traces "
            "and failure relabeling, and expose the result only as an A2 tie-breaker."
        ),
        "maps_to_current_stack": (
            "A1 supplies the live bridge diagnosis plus the goal/register predicate; "
            "A2 needs a dense value whose meaning changes when the level goal "
            "changes, instead of one global score for incompatible goals."
        ),
        "fails_when": (
            "the GOAL predicate is wrong, hindsight or failure relabeling smears "
            "incompatible level goals together, the value ignores hidden registers, "
            "or dense goal value overrides the scored-agent preservation gate."
        ),
        "roadmap_candidate": "candidate_for_v427: goal_conditioned_spatial_value_tiebreaker",
    },
]

STUDYING_SECTION_START = "<!-- EXP4625-OFFLINE-LIVE-BRIDGE-SOTA-START -->"
STUDYING_SECTION_END = "<!-- EXP4625-OFFLINE-LIVE-BRIDGE-SOTA-END -->"
STUDYING_SECTION = f"""{STUDYING_SECTION_START}
## 2026-06-23 Exp 4625 - .426 offline-live bridge SOTA ingestion - INGESTED

**Status:** INGESTED into `{NOTE_RELATIVE_PATH}`.

**Filtered track:** offline-to-live transfer for the graduated value head:
distribution shift from winning-path training to live off-path frontiers,
calibration of a ranking into a bounded cost, and compute-cost control for
value-guided search.

**Preconditions:** Hugging Face model API reachability returned `net_ok`.
`scripts/sweep_clusters.py` emitted focused value/search and ARC exploration
URLs. `scripts/sweep_semscholar.py` returned HTTP 500/429 for the five focused
queries and no S2-only source was promoted. Low-concurrency WebSearch/WebFetch
plus direct arXiv HTTP checks verified arXiv:1011.0686, arXiv:2604.11351,
arXiv:1706.04599, arXiv:2102.04518, arXiv:2406.04935, arXiv:2206.03023,
arXiv:2511.10264, and arXiv:2303.09477. `/deep-research` was not invoked.

**Methods marked ingested:** DAgger / WM-DAgger search-distribution retraining,
post-hoc value-to-cost calibration, cached decision-point Q*/limited-horizon
heuristic evaluation, SLOPE/local-heuristic bounded pruning, and
goal-conditioned offline value.

flagged_for_v427: dagger_search_distribution_value_retraining
(arXiv:1011.0686 + arXiv:2604.11351)

flagged_for_v427: calibrated_value_to_cost_tiebreaker (arXiv:1706.04599)

flagged_for_v427: decision_point_cached_qstar_value_head
(arXiv:2102.04518 + arXiv:2511.10264)

**Bottom line for .427:** first train or calibrate the value head on the live
frontier distribution, then make every live use bounded and cached; only after
those no-regression gates pass should SLOPE-style pruning or goal-conditioned
dense value affect the scored agent.
{STUDYING_SECTION_END}
"""


def build_artifact(
    *,
    methods_mapped: Sequence[JsonMap] = DEFAULT_METHODS_MAPPED,
    citations_verified: JsonMap = CITATIONS_VERIFIED,
    preconditions_checked: JsonMap = DEFAULT_PRECONDITIONS_CHECKED,
    flagged_for_next_roadmap: Sequence[str] = FLAGGED_FOR_NEXT_ROADMAP,
    honest_verdict: str = HONEST_VERDICT,
) -> dict[str, object]:
    """Build and validate the REQ-ARC-WMTE-4625 mapping artifact."""

    artifact: dict[str, object] = {
        "honest_verdict": honest_verdict,
        "inference_substrate": INFERENCE_SUBSTRATE,
        "methods_mapped": [dict(method) for method in methods_mapped],
        "flagged_for_next_roadmap": list(flagged_for_next_roadmap),
        "note_path": NOTE_RELATIVE_PATH,
        "deep_research_not_used": True,
        "preconditions_checked": dict(preconditions_checked),
        "citations_verified": {
            source_id: dict(citation) for source_id, citation in citations_verified.items()
        },
        "random_seed": RANDOM_SEED,
        "field_principles": dict(FIELD_PRINCIPLES),
    }
    validate_artifact(artifact)
    return artifact


def validate_artifact(artifact: JsonMap) -> None:
    """Validate the artifact so uncited bridge method claims fail closed."""

    missing = set(REQUIRED_ARTIFACT_FIELDS).difference(artifact)
    extra = set(artifact).difference(REQUIRED_ARTIFACT_FIELDS)
    if missing:
        raise ValueError(f"artifact missing required fields: {sorted(missing)}")
    if extra:
        raise ValueError(f"artifact has unexpected fields: {sorted(extra)}")

    verdict = artifact["honest_verdict"]
    if not isinstance(verdict, str) or not verdict.startswith(TERMINAL_PREFIXES):
        raise ValueError("honest_verdict must use a terminal prefix")
    if verdict != HONEST_VERDICT:
        raise ValueError(f"honest_verdict must equal {HONEST_VERDICT!r}")
    if artifact["inference_substrate"] != INFERENCE_SUBSTRATE:
        raise ValueError("inference_substrate must match the required substrate")
    if artifact["field_principles"] != FIELD_PRINCIPLES:
        raise ValueError("field_principles must match the required annotations")
    if artifact["note_path"] != NOTE_RELATIVE_PATH:
        raise ValueError("note_path must point at the 2026-06-23 offline-live bridge note")
    if artifact["deep_research_not_used"] is not True:
        raise ValueError("deep_research_not_used must be true")
    if artifact["random_seed"] != RANDOM_SEED:
        raise ValueError("random_seed must be the bare integer 4625")

    citations = artifact["citations_verified"]
    if not isinstance(citations, dict) or set(citations) != REQUIRED_SOURCE_IDS:
        raise ValueError("citations_verified must include exactly the required source IDs")
    for source_id, citation in citations.items():
        if not isinstance(citation, dict) or set(citation) != REQUIRED_CITATION_FIELDS:
            raise ValueError("each citation must contain exactly title, url, and http_status")
        if citation["url"] != f"https://arxiv.org/abs/{source_id}":
            raise ValueError(f"citation url must match arXiv source ID {source_id}")
        if citation["http_status"] != 200:
            raise ValueError("citation http_status must be 200")
        if not isinstance(citation["title"], str) or not citation["title"].strip():
            raise ValueError("citation title must be a non-empty string")

    methods = artifact["methods_mapped"]
    if not isinstance(methods, list) or not 3 <= len(methods) <= 5:
        raise ValueError("methods_mapped must contain three to five methods")
    for method in methods:
        if not isinstance(method, dict) or set(method) != REQUIRED_METHOD_FIELDS:
            raise ValueError("each method must contain the exact required method fields")
        source_ids = method["source_ids"]
        if not isinstance(source_ids, list) or not source_ids:
            raise ValueError("method source_ids must be a non-empty list")
        if any(source_id not in citations for source_id in source_ids):
            raise ValueError("method source_ids must all have verified citations")
        mapping = method["maps_to_current_stack"]
        if not isinstance(mapping, str) or "A1" not in mapping or "A2" not in mapping:
            raise ValueError("method mapping must name A1 and A2")
        for field in REQUIRED_METHOD_FIELDS - {"source_ids"}:
            value = method[field]
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"method {field} must be a non-empty string")

    flagged = artifact["flagged_for_next_roadmap"]
    if flagged != FLAGGED_FOR_NEXT_ROADMAP or not all(
        "flagged_for_v427" in item for item in flagged
    ):
        raise ValueError("flagged_for_next_roadmap must name the .427 candidates")

    preconditions = artifact["preconditions_checked"]
    if not isinstance(preconditions, dict) or set(preconditions) != REQUIRED_PRECONDITION_FIELDS:
        raise ValueError("preconditions_checked must contain the exact required fields")
    if preconditions["network_hf_models_reachable"] is not True:
        raise ValueError("network reachability precondition must be true")
    if preconditions["deep_research_invoked"] is not False:
        raise ValueError("deep-research must not be invoked")
    if preconditions["research_conductor_modified"] is not False:
        raise ValueError("research_conductor.py must not be modified")
    if preconditions["ops_docs_modified"] is not False:
        raise ValueError("ops docs must not be modified by Exp 4625")


def artifact_from_note(markdown: str) -> dict[str, object]:
    """Extract and validate the machine-readable JSON block from a note."""

    marker = "```json"
    start = markdown.find(marker)
    if start == -1:
        raise ValueError("research note missing machine-readable JSON block")
    json_start = markdown.find("\n", start) + 1
    json_end = markdown.find("```", json_start)
    if json_end == -1:
        raise ValueError("research note missing machine-readable JSON block terminator")
    artifact = json.loads(markdown[json_start:json_end])
    validate_artifact(artifact)
    return artifact


def validate_research_note(markdown: str) -> None:
    """Check that the paired note maps verified sources to .427 work."""

    artifact_from_note(markdown)
    required_phrases = (
        "Fresh-pass provenance",
        "SOTA -> experiment mapping",
        "A1 disambiguation",
        "A2 graduated-value-head",
        "Bottom line for the .427 roadmap",
        "flagged_for_v427",
        "DAgger",
        "WM-DAgger",
        "calibration",
        "No live LLM inference",
        "No training",
        "No leaderboard submission",
    )
    missing_phrases = [phrase for phrase in required_phrases if phrase not in markdown]
    if missing_phrases:
        raise ValueError(f"research note missing required phrase(s): {missing_phrases}")
    json_block_end = markdown.find("```\n\n## Fresh-pass provenance")
    prose = markdown[json_block_end:] if json_block_end != -1 else markdown
    missing_sources = [source for source in NOTE_REQUIRED_SOURCE_CITATIONS if source not in prose]
    if missing_sources:
        raise ValueError(f"research note missing verified source citations: {missing_sources}")


def _make_research_note(artifact: JsonMap) -> str:
    artifact_json = json.dumps(artifact, indent=2, sort_keys=True)
    return f"""# Offline-live bridge literature ingestion 2026-06-23

```json
{artifact_json}
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
"""


DEFAULT_ARTIFACT = build_artifact()
RESEARCH_NOTE = _make_research_note(DEFAULT_ARTIFACT)
validate_research_note(RESEARCH_NOTE)


def _with_studying_section(existing: str) -> str:
    if STUDYING_SECTION_START in existing and STUDYING_SECTION_END in existing:
        before, rest = existing.split(STUDYING_SECTION_START, 1)
        _, after = rest.split(STUDYING_SECTION_END, 1)
        return before.rstrip() + "\n\n" + STUDYING_SECTION.rstrip() + after
    return existing.rstrip() + "\n\n" + STUDYING_SECTION.rstrip() + "\n"


def write_outputs(
    *,
    artifact_path: Path,
    note_path: Path,
    studying_path: Path,
) -> dict[str, object]:
    """Write the JSON artifact, markdown note, and studying queue update."""

    artifact = build_artifact()
    note = _make_research_note(artifact)
    validate_research_note(note)

    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    note_path.parent.mkdir(parents=True, exist_ok=True)
    artifact_path.write_text(
        json.dumps(artifact, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    note_path.write_text(note + "\n", encoding="utf-8")
    studying_path.write_text(
        _with_studying_section(studying_path.read_text(encoding="utf-8")),
        encoding="utf-8",
    )
    return artifact


def main() -> int:
    """Write the default Exp 4625 deliverables under the repository root."""

    repo_root = Path(os.environ.get("CARNOT_EXP4625_ROOT", Path(__file__).resolve().parents[2]))
    artifact = write_outputs(
        artifact_path=repo_root / RESULT_RELATIVE_PATH,
        note_path=repo_root / NOTE_RELATIVE_PATH,
        studying_path=repo_root / "research-studying.md",
    )
    print(artifact["honest_verdict"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
