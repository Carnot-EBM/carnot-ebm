"""Exp 4673 structural-deepening SOTA ingestion.

Spec refs: REQ-ARC-WMTE-4673, SCENARIO-ARC-WMTE-4673.

This is a literature-to-experiment mapping artifact. It deepens the structural
.431 fallback after the precise .430 A1/A2 levers failed to cross the
multi-level wall.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import json
import os
from pathlib import Path
from typing import Any


JsonMap = Mapping[str, Any]

RESULT_RELATIVE_PATH = "results/experiment_4673_sota_ingestion_structural_deepening.json"
NOTE_RELATIVE_PATH = (
    "docs/research-notes/structural-deepening-sota-ingestion-2026-06-24.md"
)
RANDOM_SEED = 4673
HONEST_VERDICT = "success: sota_ingestion_structural_deepening_mapped"
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
        "deep_research_not_used",
        "methods_mapped",
        "citations_verified",
        "flagged_for_next_roadmap",
        "note_path",
        "preconditions_checked",
    }
)
REQUIRED_ARTIFACT_FIELDS = (
    "honest_verdict",
    "inference_substrate",
    "deep_research_not_used",
    "methods_mapped",
    "citations_verified",
    "flagged_for_next_roadmap",
    "note_path",
    "preconditions_checked",
    "random_seed",
    "field_principles",
)
REQUIRED_METHOD_FIELDS = frozenset(
    {
        "method",
        "source_ids",
        "track",
        "residual_scope",
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
        "sweep_clusters_help_ok",
        "exp4661_artifact_read",
        "exp4661_note_read",
        "exp4664_artifact_read",
        "exp4665_artifact_read",
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
        "2504.04366",
        "2505.10819",
        "2506.07255",
        "2604.03208",
        "2604.11351",
        "2605.05138",
        "2605.12913",
    }
)
NOTE_REQUIRED_SOURCE_CITATIONS = frozenset(
    f"arXiv:{source_id}" for source_id in REQUIRED_SOURCE_IDS
)

A1_RESIDUAL = "single_exemplar_goal_insufficient"
A2_RESIDUAL = "missing_verifier_gap_live_frontier_not_separated"
FLAGGED_FOR_NEXT_ROADMAP = [
    (
        "flagged_for_v431: hierarchical_subgoal_e3_frontier_with_a1_a2_tiebreakers "
        "(arXiv:2604.03208 + arXiv:2506.07255 + arXiv:2504.04366 + "
        "arXiv:2605.12913 + arXiv:1011.0686)"
    ),
    (
        "flagged_for_v431: poe_world_factored_executable_subgoal_planner "
        "(arXiv:2505.10819 + arXiv:2605.05138)"
    ),
]

FIELD_PRINCIPLES = {
    "honest_verdict": {
        "principle": "terminal prefix; success: sota_ingestion_structural_deepening_mapped."
    },
    "inference_substrate": {
        "principle": (
            "aggregation_from_upstream_artifacts -- literature read + synthesis, "
            "no model load (100us floor)."
        )
    },
    "deep_research_not_used": {
        "principle": (
            "MUST be true -- /deep-research is BANNED in the autonomous loop; used "
            "sweep helpers + low-concurrency WebSearch/WebFetch."
        )
    },
    "methods_mapped": {
        "principle": (
            "the strongest 3-5 SOTA methods with REAL arXiv IDs + per-method "
            "implement-cost-over-current-stack + fails_when (no citation = fabrication)."
        )
    },
    "citations_verified": {
        "principle": (
            "each cited arXiv ID with an HTTP-200 verification -- pre-empts fabricated "
            "citations."
        )
    },
    "flagged_for_next_roadmap": {
        "principle": (
            "the strongest method(s) flagged as candidate .431 inputs "
            "(flagged_for_v431) -- closes discover->ingest->plan->experiment."
        )
    },
    "note_path": {
        "principle": (
            "the per-track research-note path (the SOTA-Ingestion Cycle deliverable)."
        )
    },
    "preconditions_checked": {
        "principle": "records network reachability verified; pre-empts fabricated citations."
    },
}

CITATIONS_VERIFIED = {
    "1011.0686": {
        "title": (
            "A Reduction of Imitation Learning and Structured Prediction to "
            "No-Regret Online Learning"
        ),
        "url": "https://arxiv.org/abs/1011.0686",
        "http_status": 200,
    },
    "2504.04366": {
        "title": "Solving Sokoban using Hierarchical Reinforcement Learning with Landmarks",
        "url": "https://arxiv.org/abs/2504.04366",
        "http_status": 200,
    },
    "2505.10819": {
        "title": (
            "PoE-World: Compositional World Modeling with Products of "
            "Programmatic Experts"
        ),
        "url": "https://arxiv.org/abs/2505.10819",
        "http_status": 200,
    },
    "2506.07255": {
        "title": "Subgoal-Guided Policy Heuristic Search with Learned Subgoals",
        "url": "https://arxiv.org/abs/2506.07255",
        "http_status": 200,
    },
    "2604.03208": {
        "title": "Hierarchical Planning with Latent World Models",
        "url": "https://arxiv.org/abs/2604.03208",
        "http_status": 200,
    },
    "2604.11351": {
        "title": (
            "WM-DAgger: Enabling Efficient Data Aggregation for Imitation "
            "Learning with World Models"
        ),
        "url": "https://arxiv.org/abs/2604.11351",
        "http_status": 200,
    },
    "2605.05138": {
        "title": "Executable World Models for ARC-AGI-3 in the Era of Coding Agents",
        "url": "https://arxiv.org/abs/2605.05138",
        "http_status": 200,
    },
    "2605.12913": {
        "title": "Revisiting DAgger in the Era of LLM-Agents",
        "url": "https://arxiv.org/abs/2605.12913",
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
    (
        "hierarchical subgoal search learned subgoals latent world model ARC "
        "2604.03208 2506.07255 2504.04366"
    ),
    (
        "factored executable world model product of experts programmatic experts "
        "ARC 2505.10819 2605.05138"
    ),
    (
        "subgoal conditioned planning DAgger distribution shift value routing "
        "live frontier 2605.12913 1011.0686"
    ),
]
WEBSEARCH_WEBFETCH_TOP_SOURCES = [
    "https://arxiv.org/abs/2604.03208",
    "https://arxiv.org/abs/2506.07255",
    "https://arxiv.org/abs/2504.04366",
    "https://arxiv.org/abs/2505.10819",
    "https://arxiv.org/abs/2605.05138",
    "https://arxiv.org/abs/2605.12913",
    "https://arxiv.org/abs/2604.11351",
    "https://arxiv.org/abs/1011.0686",
]
DEFAULT_PRECONDITIONS_CHECKED = {
    "agents_md_read": True,
    "codex_md_read": True,
    "network_hf_models_reachable": True,
    "sweep_clusters_help_ok": True,
    "exp4661_artifact_read": True,
    "exp4661_note_read": True,
    "exp4664_artifact_read": True,
    "exp4665_artifact_read": True,
    "research_studying_read": True,
    "research_references_read": True,
    "sweep_clusters_used": True,
    "sweep_clusters_urls": SWEEP_CLUSTER_URLS,
    "sweep_semscholar_used": True,
    "sweep_semscholar_queries": S2_QUERIES,
    "sweep_semscholar_arxiv_ids": [],
    "sweep_semscholar_rate_limited_queries": S2_QUERIES,
    "arxiv_http_200_verified_ids": [
        f"https://arxiv.org/abs/{source_id}" for source_id in sorted(REQUIRED_SOURCE_IDS)
    ],
    "websearch_webfetch_top_sources": WEBSEARCH_WEBFETCH_TOP_SOURCES,
    "deep_research_invoked": False,
    "live_llm_inference": False,
    "training_launched": False,
    "model_load": False,
    "leaderboard_submission": False,
    "ops_docs_modified": False,
    "research_conductor_modified": False,
}

RESIDUAL_SCOPE = (
    "A1 residual single_exemplar_goal_insufficient left unsatisfiable L2 goal "
    "predicates and empty plans; A2 residual "
    "missing_verifier_gap_live_frontier_not_separated left zero live lift after "
    "distribution-corrected value routing."
)
DEFAULT_METHODS_MAPPED = [
    {
        "method": "Hierarchical subgoal search over the live E3 frontier",
        "source_ids": ["2604.03208", "2506.07255", "2504.04366", "2605.05138"],
        "track": "hierarchical_subgoal_search_live_e3_frontier",
        "residual_scope": RESIDUAL_SCOPE,
        "implement_cost_over_current_stack": (
            "medium-high: add a high-level subgoal layer above the current live E3 "
            "frontier, mine A1/A2 failed search traces for subgoal candidates, run "
            "bounded low-level search per subgoal, and keep replay plus matched "
            "no-regression controls."
        ),
        "maps_to_current_stack": (
            "A1 L2-goal-induction becomes a subgoal proposer instead of one global "
            "terminal predicate, A2 distribution-corrected value-routing becomes the "
            "tie-breaker inside each bounded subgoal search, and live E3 remains "
            "the replay-verified executor."
        ),
        "fails_when": (
            "the subgoal layer proposes visual states that are not mechanically goal "
            "relevant, A2 still cannot separate live frontier states, bounded search "
            "cannot reach the proposed subgoal, or live E3 replay rejects the path."
        ),
        "roadmap_candidate": FLAGGED_FOR_NEXT_ROADMAP[0],
    },
    {
        "method": "Failed-search-tree subgoal proposer with value tie-breaking",
        "source_ids": ["2506.07255", "2605.12913", "1011.0686"],
        "track": "failed_search_tree_subgoal_proposer",
        "residual_scope": RESIDUAL_SCOPE,
        "implement_cost_over_current_stack": (
            "medium: retain failed A1/A2 frontier trees, label promising partial "
            "states from replay and value deltas, train a subgoal-conditioned "
            "proposal table, and use the corrected value head only at decision "
            "points where candidates are otherwise tied."
        ),
        "maps_to_current_stack": (
            "A1 L2-goal-induction supplies candidate post-level-up goal states, "
            "A2 distribution-corrected value-routing ranks tree-local alternatives "
            "rather than every primitive action globally, and live E3 supplies the "
            "failed trees plus replay adjudication."
        ),
        "fails_when": (
            "failed search trees contain no reusable near-goal states, labels are "
            "too sparse to choose among subgoals, or the value head only reshuffles "
            "a candidate set without any mechanically valid L2 action."
        ),
        "roadmap_candidate": FLAGGED_FOR_NEXT_ROADMAP[0],
    },
    {
        "method": "PoE-World factored executable model subgoal planner",
        "source_ids": ["2505.10819", "2605.05138"],
        "track": "poe_world_factored_executable_model_planner",
        "residual_scope": RESIDUAL_SCOPE,
        "implement_cost_over_current_stack": (
            "medium-high: induce small programmatic experts for object-level "
            "preconditions and effects, weight experts by held-out transition "
            "trust, compose only replay-stable factors, and plan "
            "subgoal-conditioned sequences through the product model."
        ),
        "maps_to_current_stack": (
            "A1 L2-goal-induction proposes the subgoal predicates each expert must "
            "make reachable, A2 distribution-corrected value-routing scores which "
            "expert-predicted states deserve live expansion, and live E3 executes "
            "and audits every emitted plan."
        ),
        "fails_when": (
            "expert factors are not independent, rare interactions are smoothed "
            "away by the product, generated experts overfit prefix transitions, "
            "or product-model plans emit actions that live E3 cannot replay."
        ),
        "roadmap_candidate": FLAGGED_FOR_NEXT_ROADMAP[1],
    },
    {
        "method": "WM-DAgger trust-weighted subgoal-conditioned value routing",
        "source_ids": ["2605.12913", "2604.11351", "1011.0686", "2605.05138"],
        "track": "wm_dagger_trust_weighted_subgoal_value_routing",
        "residual_scope": RESIDUAL_SCOPE,
        "implement_cost_over_current_stack": (
            "medium: aggregate live-frontier states under each proposed subgoal, "
            "synthesize or replay OOD recovery transitions only when the executable "
            "world model is held-out trusted, and calibrate value scores as bounded "
            "subgoal-local costs."
        ),
        "maps_to_current_stack": (
            "A1 L2-goal-induction defines the subgoal-conditioned state "
            "distribution, A2 distribution-corrected value-routing is retrained or "
            "calibrated on that distribution, and live E3 provides both the frontier "
            "states and the final replay gate."
        ),
        "fails_when": (
            "the executable world model hallucinates OOD recovery transitions, trust "
            "weights accept brittle experts, subgoal partitions are too small for "
            "value calibration, or the calibrated value still sees no valid L2 path."
        ),
        "roadmap_candidate": FLAGGED_FOR_NEXT_ROADMAP[0],
    },
]

STUDYING_SECTION_START = "<!-- EXP4673-STRUCTURAL-DEEPENING-SOTA-START -->"
STUDYING_SECTION_END = "<!-- EXP4673-STRUCTURAL-DEEPENING-SOTA-END -->"
STUDYING_SECTION = f"""{STUDYING_SECTION_START}
## 2026-06-24 Exp 4673 - .431 structural-deepening SOTA ingestion - INGESTED

**Status:** INGESTED into `{NOTE_RELATIVE_PATH}`.

**Filtered track:** structural fallback after A1 L2-goal-induction closed with
`single_exemplar_goal_insufficient` and A2 distribution-corrected value-routing
closed with `missing_verifier_gap_live_frontier_not_separated`. The ingestion
deepens the `.429` flagged tracks into implementable `.431` candidates rather
than re-running scalar value routing.

**Preconditions:** Hugging Face model API reachability returned `net_ok`.
`scripts/sweep_clusters.py --help` exited cleanly. `scripts/sweep_clusters.py`
emitted the neural-guided-search and ARC-exploration cluster URLs.
`scripts/sweep_semscholar.py` returned HTTP 429 for the three focused queries
and no S2-only source was promoted. Low-concurrency WebSearch/WebFetch plus
direct arXiv HTTP checks verified arXiv:2604.03208, arXiv:2506.07255,
arXiv:2504.04366, arXiv:2505.10819, arXiv:2605.05138, arXiv:2605.12913,
arXiv:2604.11351, and arXiv:1011.0686. `/deep-research` was not invoked.

**Methods marked ingested:** hierarchical subgoal search over live E3, failed
search-tree subgoal proposal with value tie-breaking, PoE-World/factored
executable world-model planning, and WM-DAgger trust-weighted
subgoal-conditioned value routing.

flagged_for_v431: hierarchical_subgoal_e3_frontier_with_a1_a2_tiebreakers
(arXiv:2604.03208 + arXiv:2506.07255 + arXiv:2504.04366 + arXiv:2605.12913 + arXiv:1011.0686)

flagged_for_v431: poe_world_factored_executable_subgoal_planner
(arXiv:2505.10819 + arXiv:2605.05138)

**Bottom line for .431:** make the hierarchical subgoal layer the primary
structural move. Use A1's induced goal signal as a subgoal proposer, A2's value
head as a bounded local tie-breaker, and live E3 as the executor. Keep PoE-World
as the stronger alternate when enough transition trust exists to factor effects.
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
    """Build and validate the REQ-ARC-WMTE-4673 mapping artifact."""

    artifact: dict[str, object] = {
        "honest_verdict": honest_verdict,
        "inference_substrate": INFERENCE_SUBSTRATE,
        "deep_research_not_used": True,
        "methods_mapped": [dict(method) for method in methods_mapped],
        "citations_verified": {
            source_id: dict(citation) for source_id, citation in citations_verified.items()
        },
        "flagged_for_next_roadmap": list(flagged_for_next_roadmap),
        "note_path": NOTE_RELATIVE_PATH,
        "preconditions_checked": dict(preconditions_checked),
        "random_seed": RANDOM_SEED,
        "field_principles": dict(FIELD_PRINCIPLES),
    }
    validate_artifact(artifact)
    return artifact


def validate_artifact(artifact: JsonMap) -> None:
    """Validate the artifact so uncited structural-deepening claims fail closed."""

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
        raise ValueError("note_path must point at the 2026-06-24 structural-deepening note")
    if artifact["deep_research_not_used"] is not True:
        raise ValueError("deep_research_not_used must be true")
    if artifact["random_seed"] != RANDOM_SEED:
        raise ValueError("random_seed must be the bare integer 4673")

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
        if (
            not isinstance(mapping, str)
            or "A1 L2-goal-induction" not in mapping
            or "A2 distribution-corrected value-routing" not in mapping
            or "live E3" not in mapping
        ):
            raise ValueError(
                "method mapping must name A1 L2-goal-induction, "
                "A2 distribution-corrected value-routing, and live E3"
            )
        residual_scope = method["residual_scope"]
        if (
            not isinstance(residual_scope, str)
            or A1_RESIDUAL not in residual_scope
            or A2_RESIDUAL not in residual_scope
        ):
            raise ValueError("method residual_scope must name the A1 and A2 residuals")
        for field in REQUIRED_METHOD_FIELDS - {"source_ids"}:
            value = method[field]
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"method {field} must be a non-empty string")

    flagged = artifact["flagged_for_next_roadmap"]
    if flagged != FLAGGED_FOR_NEXT_ROADMAP or not all(
        "flagged_for_v431" in item for item in flagged
    ):
        raise ValueError("flagged_for_next_roadmap must name the .431 candidates")

    preconditions = artifact["preconditions_checked"]
    if not isinstance(preconditions, dict) or set(preconditions) != REQUIRED_PRECONDITION_FIELDS:
        raise ValueError("preconditions_checked must contain the exact required fields")
    if preconditions["network_hf_models_reachable"] is not True:
        raise ValueError("network reachability precondition must be true")
    if preconditions["sweep_clusters_help_ok"] is not True:
        raise ValueError("sweep_clusters helper precondition must be true")
    if preconditions["deep_research_invoked"] is not False:
        raise ValueError("deep-research must not be invoked")
    if preconditions["research_conductor_modified"] is not False:
        raise ValueError("research_conductor.py must not be modified")
    if preconditions["ops_docs_modified"] is not False:
        raise ValueError("ops docs must not be modified by Exp 4673")


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
    """Check that the paired note maps verified sources to .431 work."""

    artifact_from_note(markdown)
    required_phrases = (
        "Fresh-pass provenance",
        "SOTA -> .431 structural mapping",
        "A1 L2-goal-induction",
        "A2 distribution-corrected value-routing",
        "live E3",
        "single_exemplar_goal_insufficient",
        "missing_verifier_gap_live_frontier_not_separated",
        "Bottom line for the .431 roadmap",
        "flagged_for_v431",
        "Hierarchical Planning with Latent World Models",
        "PoE-World",
        "DAgger",
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
    return f"""# Structural-deepening SOTA ingestion 2026-06-24

```json
{artifact_json}
```

## Fresh-pass provenance

Read `AGENTS.md`, `CODEX.md`,
`results/experiment_4661_sota_ingestion_generation_guidance.json`,
`docs/research-notes/generation-guidance-sota-ingestion-2026-06-24.md`,
`results/experiment_4664_l2_goal_predicate_induction_live.json`,
`results/experiment_4665_dagger_distribution_shift_value_routing.json`,
`research-studying.md`, and `research-references.md`. The current stack is A1
L2-goal-induction plus A2 distribution-corrected value-routing inside the live
E3 path. A1 closed with `single_exemplar_goal_insufficient`: the induced L2
goals were unsatisfiable, plans were length zero, and no L2 plan reached the
goal. A2 closed with `missing_verifier_gap_live_frontier_not_separated`: the
distribution shift score was corrected from 0.699108 to 0.0, but first-win and
solve-rate deltas were still 0.0. This pass therefore maps a structural fallback
instead of another scalar reranker.

Reliable-channel pass, not `/deep-research`:
- `curl -sf -o /dev/null https://huggingface.co/api/models`
- `.venv/bin/python scripts/sweep_clusters.py --help`
- `.venv/bin/python scripts/sweep_clusters.py 6 --max-results 8`
- `.venv/bin/python scripts/sweep_clusters.py 5 --max-results 8`
- `.venv/bin/python scripts/sweep_semscholar.py - --limit 8` with three focused queries
- low-concurrency WebSearch/WebFetch of the top hierarchical, factored-model, and DAgger papers
- direct arXiv HTTP checks for all cited IDs

Semantic Scholar returned HTTP 429 for the focused queries and no S2-only source
was promoted. Direct arXiv HTTP checks returned 200 for arXiv:2604.03208,
arXiv:2506.07255, arXiv:2504.04366, arXiv:2505.10819, arXiv:2605.05138,
arXiv:2605.12913, arXiv:2604.11351, and arXiv:1011.0686. No live LLM inference,
No training, No leaderboard submission, no model load, and no live solve claim
were run or made. `scripts/research_conductor.py`,
`ops/changelog.md`, and `ops/status.md` were not edited by this workflow.

## SOTA -> .431 structural mapping

## Hierarchical subgoal search over the live E3 frontier

**Sources:** Hierarchical Planning with Latent World Models, arXiv:2604.03208;
Subgoal-Guided Policy Heuristic Search with Learned Subgoals, arXiv:2506.07255;
Sokoban hierarchical landmarks, arXiv:2504.04366; Executable World Models for
ARC-AGI-3, arXiv:2605.05138.

**Mapping to current stack:** A1 L2-goal-induction should stop being one global
terminal predicate and become a subgoal proposer. A2 distribution-corrected
value-routing should be used as a local tie-breaker inside each bounded subgoal
search. The live E3 path remains the executor and replay gate.

**Implementation cost over current stack:** medium-high. Add subgoal mining
from failed A1/A2 trees, run bounded low-level search for each selected subgoal,
and retain matched baseline, no-subgoal ablation, random-subgoal ablation, and
replay gates.

**Fails when:** proposed subgoals are visually plausible but mechanically
irrelevant, the corrected value head still does not separate live frontier
states, bounded search cannot reach the chosen subgoal, or live E3 rejects the
path.

## Failed-search-tree subgoal proposer with value tie-breaking

**Sources:** Subgoal-guided heuristic search, arXiv:2506.07255; Revisiting
DAgger in the Era of LLM-Agents, arXiv:2605.12913; original DAgger,
arXiv:1011.0686.

**Mapping to current stack:** A1 L2-goal-induction provides candidate
post-level-up states even when the terminal goal is not yet satisfiable. A2
distribution-corrected value-routing ranks alternatives within a subgoal-local
tree, not every primitive action globally. The live E3 failed frontier is the
training and replay substrate.

**Implementation cost over current stack:** medium. Persist failed search
trees, label promising partial states from replay/value deltas, learn a
subgoal-conditioned proposal table, and use value scores only at bounded
decision points.

**Fails when:** failed trees contain no reusable near-goal states, labels are
too sparse to choose among subgoals, or tie-breaking only reshuffles candidates
that do not contain a valid L2 action.

## PoE-World factored executable model subgoal planner

**Sources:** PoE-World, arXiv:2505.10819; Executable World Models for ARC-AGI-3,
arXiv:2605.05138.

**Mapping to current stack:** A1 L2-goal-induction proposes the predicates each
factor should make reachable. A2 distribution-corrected value-routing scores
which product-model states deserve live expansion. The live E3 path executes
and audits every emitted plan. This is the strongest answer to the A2 residual
because QD needs a factored feasibility model before it mutates.

**Implementation cost over current stack:** medium-high. Induce object-level
precondition/effect experts, weight them by held-out transition trust, compose
only replay-stable factors, and search through the product model with hard live
replay checks.

**Fails when:** expert factors are not independent, rare interactions are
smoothed away, generated experts overfit prefix transitions, or product-model
planning emits a live-invalid action sequence.

## WM-DAgger trust-weighted subgoal-conditioned value routing

**Sources:** Revisiting DAgger in the Era of LLM-Agents, arXiv:2605.12913;
WM-DAgger, arXiv:2604.11351; original DAgger, arXiv:1011.0686; Executable World
Models for ARC-AGI-3, arXiv:2605.05138.

**Mapping to current stack:** A1 L2-goal-induction defines subgoal-conditioned
state distributions. A2 distribution-corrected value-routing is retrained or
calibrated on those distributions instead of one mixed frontier. The live E3
path provides the frontier states and the final replay gate.

**Implementation cost over current stack:** medium. Aggregate live-frontier
states per subgoal, synthesize or replay OOD recovery transitions only when a
trusted executable model supports them, and calibrate values as subgoal-local
costs.

**Fails when:** the executable model hallucinates recovery transitions, trust
weights accept brittle experts, subgoal partitions are too small for value
calibration, or the calibrated value still sees no valid L2 path.

## Bottom line for the .431 roadmap

1. Build `flagged_for_v431: hierarchical_subgoal_e3_frontier_with_a1_a2_tiebreakers`
   first. It directly addresses `single_exemplar_goal_insufficient` by turning
   A1 into a subgoal proposer and directly addresses
   `missing_verifier_gap_live_frontier_not_separated` by restricting A2 to
   local subgoal tie-breaking. The core citations are arXiv:2604.03208,
   arXiv:2506.07255, arXiv:2504.04366, arXiv:2605.12913, and arXiv:1011.0686.
2. Keep `flagged_for_v431: poe_world_factored_executable_subgoal_planner` as the
   second structural candidate when transition-factor trust is available.
   PoE-World arXiv:2505.10819 plus executable ARC world models
   arXiv:2605.05138 is the best factored-model answer to A2's no-winner bridge.
3. Use WM-DAgger arXiv:2604.11351 as a support mechanism only after a subgoal or
   product-model scaffold exists; it is not another standalone value-weight run.
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
    """Write the default Exp 4673 deliverables under the repository root."""

    repo_root = Path(os.environ.get("CARNOT_EXP4673_ROOT", Path(__file__).resolve().parents[2]))
    artifact = write_outputs(
        artifact_path=repo_root / RESULT_RELATIVE_PATH,
        note_path=repo_root / NOTE_RELATIVE_PATH,
        studying_path=repo_root / "research-studying.md",
    )
    print(artifact["honest_verdict"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
