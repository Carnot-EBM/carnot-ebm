"""Exp 4661 generation-guidance SOTA ingestion.

Spec refs: REQ-ARC-WMTE-4661, SCENARIO-ARC-WMTE-4661.

This is a literature-to-experiment mapping artifact, not a benchmark run. It
records the 2026-06-24 focused pass over surviving generation-guidance methods
after this week's macro-action, click-heatmap, just-explore schedule, and
standalone goal-energy heuristic levers were retired or nulled.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import json
import os
from pathlib import Path
from typing import Any


JsonMap = Mapping[str, Any]

RESULT_RELATIVE_PATH = "results/experiment_4661_sota_ingestion_generation_guidance.json"
NOTE_RELATIVE_PATH = "docs/research-notes/generation-guidance-sota-ingestion-2026-06-24.md"
RANDOM_SEED = 4661
HONEST_VERDICT = "success: sota_ingestion_generation_guidance_mapped"
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
        "dead_levers_not_reflagged",
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
    "dead_levers_not_reflagged",
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
        "exp4649_artifact_read",
        "exp4649_note_read",
        "research_studying_read",
        "research_references_read",
        "dead_lever_notes_read",
        "a1_value_routing_artifact_read",
        "a2_energy_fitness_qd_artifact_read",
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
        "1706.04599",
        "2102.04518",
        "2308.05483",
        "2504.01915",
        "2504.04366",
        "2505.10819",
        "2506.07255",
        "2604.03208",
        "2604.11351",
        "2605.05138",
        "2605.28814",
    }
)
NOTE_REQUIRED_SOURCE_CITATIONS = frozenset(
    f"arXiv:{source_id}" for source_id in REQUIRED_SOURCE_IDS
)
DEAD_LEVERS_NOT_REFLAGGED = [
    "macro-action horizon-collapse RETIRED",
    "click-heatmap off-centroid generator RETIRED",
    "just-explore schedule-extraction CLOSED",
    "goal-energy heuristic NULL",
]
DEAD_REFLAG_TOKENS = (
    "macro_action_horizon_collapse",
    "macro-action horizon-collapse",
    "click_heatmap",
    "click-heatmap",
    "just_explore_schedule",
    "just-explore schedule",
    "goal_energy_heuristic",
    "goal-energy heuristic",
)
FLAGGED_FOR_NEXT_ROADMAP = [
    (
        "flagged_for_v430: hierarchical_subgoal_e3_frontier_with_distribution_shift_value_routing "
        "(arXiv:2604.03208 + arXiv:2506.07255 + arXiv:2504.04366 + "
        "arXiv:1011.0686 + arXiv:2604.11351 + arXiv:1706.04599)"
    ),
    (
        "flagged_for_v430: poe_world_factored_executable_subgoal_planner "
        "(arXiv:2505.10819 + arXiv:2605.05138)"
    ),
]

FIELD_PRINCIPLES = {
    "honest_verdict": {
        "principle": "terminal prefix; success: sota_ingestion_generation_guidance_mapped."
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
            "the strongest method(s) flagged as candidate .430 inputs "
            "(flagged_for_v430) -- closes discover->ingest->plan->experiment."
        )
    },
    "dead_levers_not_reflagged": {
        "principle": (
            "names the DEAD levers (macro/click/schedule/goal-energy heuristic) "
            "confirmed NOT re-flagged -- honors the week's falsifications."
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
    "2308.05483": {
        "title": "Quality Diversity under Sparse Reward and Sparse Interaction",
        "url": "https://arxiv.org/abs/2308.05483",
        "http_status": 200,
    },
    "2504.01915": {
        "title": (
            "Overcoming Deceptiveness in Fitness Optimization with Unsupervised "
            "Quality-Diversity"
        ),
        "url": "https://arxiv.org/abs/2504.01915",
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
        "title": "WM-DAgger: Enabling Efficient Data Aggregation for Imitation Learning",
        "url": "https://arxiv.org/abs/2604.11351",
        "http_status": 200,
    },
    "2605.05138": {
        "title": "Executable World Models for ARC-AGI-3 in the Era of Coding Agents",
        "url": "https://arxiv.org/abs/2605.05138",
        "http_status": 200,
    },
    "2605.28814": {
        "title": "Self-Improving Language Models with Bidirectional Evolutionary Search",
        "url": "https://arxiv.org/abs/2605.28814",
        "http_status": 200,
    },
}

SWEEP_CLUSTER_URLS = [
    (
        "http://export.arxiv.org/api/query?search_query="
        '(abs:"affordance"+OR+abs:"action+effect"+OR+abs:"clickability"+OR+'
        'abs:"frame+prediction"+OR+abs:"intrinsic+motivation"+OR+'
        'abs:"directed+exploration"+OR+abs:"novelty+search")+AND+'
        '(abs:"reinforcement+learning"+OR+abs:"agent"+OR+abs:"exploration"+OR+'
        'abs:"interactive+environment"+OR+abs:"ARC")&start=0&max_results=8'
        "&sortBy=submittedDate&sortOrder=descending"
    ),
    (
        "http://export.arxiv.org/api/query?search_query="
        '(abs:"neural+guided+search"+OR+abs:"learned+heuristic"+OR+'
        'abs:"value+guided+search"+OR+abs:"program+induction"+OR+'
        'abs:"world+model"+OR+abs:"goal+induction")+AND+'
        '(abs:"planning"+OR+abs:"agent"+OR+abs:"reasoning"+OR+'
        'abs:"reinforcement+learning")&start=0&max_results=8'
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
        "distribution shift corrected value routing affordable value routing "
        "offline to live 1011.0686 2604.11351 1706.04599 2102.04518"
    ),
]
WEBSEARCH_WEBFETCH_TOP_SOURCES = [
    "https://arxiv.org/abs/2604.03208",
    "https://arxiv.org/abs/2506.07255",
    "https://arxiv.org/abs/2504.04366",
    "https://arxiv.org/abs/2505.10819",
    "https://arxiv.org/abs/2605.05138",
    "https://arxiv.org/abs/1011.0686",
    "https://arxiv.org/abs/2604.11351",
    "https://arxiv.org/abs/1706.04599",
]

DEFAULT_PRECONDITIONS_CHECKED = {
    "agents_md_read": True,
    "codex_md_read": True,
    "network_hf_models_reachable": True,
    "sweep_clusters_help_ok": True,
    "exp4649_artifact_read": True,
    "exp4649_note_read": True,
    "research_studying_read": True,
    "research_references_read": True,
    "dead_lever_notes_read": DEAD_LEVERS_NOT_REFLAGGED,
    "a1_value_routing_artifact_read": True,
    "a2_energy_fitness_qd_artifact_read": True,
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

DEFAULT_METHODS_MAPPED = [
    {
        "method": "Hierarchical subgoal search over the live E3 frontier",
        "source_ids": ["2604.03208", "2506.07255", "2504.04366", "2605.05138"],
        "track": "hierarchical_subgoal_search_live_e3_frontier",
        "implement_cost_over_current_stack": (
            "medium: add a high-level subgoal layer above the current live E3 "
            "frontier, mine failed A1/A2 search trees for subgoal candidates, "
            "route each subgoal through bounded low-level search, and keep "
            "replay checks plus matched no-regression controls."
        ),
        "maps_to_current_stack": (
            "A1 value-routing supplies a calibrated low-cost tie-breaker inside "
            "each subgoal search, A2 energy-fitness QD becomes a subgoal-conditioned "
            "sequence proposer rather than a standalone archive, and live E3 remains "
            "the replay-verified executor and parity surface."
        ),
        "fails_when": (
            "the subgoal miner proposes visual states that are not goal relevant, "
            "value routing is still shifted from live frontier states, QD spends "
            "budget around an unreachable landmark, or live E3 replay rejects the "
            "subgoal path."
        ),
        "roadmap_candidate": FLAGGED_FOR_NEXT_ROADMAP[0],
    },
    {
        "method": "PoE-World factored executable model subgoal planner",
        "source_ids": ["2505.10819", "2605.05138"],
        "track": "poe_world_factored_executable_model_planner",
        "implement_cost_over_current_stack": (
            "medium-high: induce small programmatic experts for object-level "
            "preconditions and effects, weight them by held-out transition trust, "
            "compose only replay-stable factors, and plan subgoal-conditioned "
            "candidate sequences through the product model."
        ),
        "maps_to_current_stack": (
            "A1 value-routing scores which expert-predicted states deserve live "
            "expansion, A2 energy-fitness QD mutates only sequences that the "
            "factored executable model marks feasible, and live E3 adjudicates "
            "every emitted plan through its normal action/replay path."
        ),
        "fails_when": (
            "expert factors are not independent, rare interactions are smoothed "
            "away by the product, generated experts overfit prefix transitions, "
            "or the soft plan yields actions that live E3 cannot replay."
        ),
        "roadmap_candidate": FLAGGED_FOR_NEXT_ROADMAP[1],
    },
    {
        "method": "Distribution-shift-corrected value routing for subgoal frontiers",
        "source_ids": ["1011.0686", "2604.11351", "1706.04599", "2102.04518"],
        "track": "distribution_shift_corrected_value_routing",
        "implement_cost_over_current_stack": (
            "medium: collect live-frontier states from A1 and A2 failures, use "
            "DAgger-style aggregation to retrain or recalibrate the value router "
            "on off-path states, convert scores to bounded cost deltas, and cache "
            "decision-point evaluations so routing stays affordable."
        ),
        "maps_to_current_stack": (
            "A1 value-routing stops applying a winning-path value head to shifted "
            "live frontier states, A2 energy-fitness QD receives calibrated "
            "subgoal costs instead of raw goal-energy ranking, and live E3 keeps "
            "primitive actions plus parity gates as the scored integration point."
        ),
        "fails_when": (
            "the aggregated frontier data is too small, calibration collapses "
            "under hidden-state games, cached Q-values become stale after level-up, "
            "or the router only reorders candidates that A2 and live E3 still fail "
            "to generate."
        ),
        "roadmap_candidate": FLAGGED_FOR_NEXT_ROADMAP[0],
    },
]

STUDYING_SECTION_START = "<!-- EXP4661-GENERATION-GUIDANCE-SOTA-START -->"
STUDYING_SECTION_END = "<!-- EXP4661-GENERATION-GUIDANCE-SOTA-END -->"
STUDYING_SECTION = f"""{STUDYING_SECTION_START}
## 2026-06-24 Exp 4661 - .429 generation-guidance SOTA ingestion - INGESTED

**Status:** INGESTED into `{NOTE_RELATIVE_PATH}`.

**Filtered track:** surviving generation-guidance directions for chaining a
second live level-up after A1 value-routing and A2 energy-fitness QD both
returned no live lift: hierarchical subgoal search, PoE/factored executable
world models, and distribution-shift-corrected value routing.

**Preconditions:** Hugging Face model API reachability returned `net_ok`.
`scripts/sweep_clusters.py --help` exited cleanly. `scripts/sweep_clusters.py`
emitted focused ARC exploration and neural-guided-search URLs.
`scripts/sweep_semscholar.py` returned HTTP 429 for the three focused queries
and no S2-only source was promoted. Low-concurrency WebSearch/WebFetch plus
direct arXiv HTTP checks verified arXiv:2604.03208, arXiv:2506.07255,
arXiv:2504.04366, arXiv:2505.10819, arXiv:2605.05138, arXiv:1011.0686,
arXiv:2604.11351, arXiv:1706.04599, arXiv:2102.04518, arXiv:2605.28814,
arXiv:2308.05483, and arXiv:2504.01915. `/deep-research` was not invoked.

**Dead levers confirmed not re-flagged:** macro-action horizon-collapse
RETIRED; click-heatmap off-centroid generator RETIRED; just-explore
schedule-extraction CLOSED; goal-energy heuristic NULL.

**Methods marked ingested:** hierarchical subgoal search over live E3,
PoE-World/factored executable world-model planning, and DAgger/calibrated
distribution-shift-corrected value routing for subgoal frontiers.

flagged_for_v430: hierarchical_subgoal_e3_frontier_with_distribution_shift_value_routing
(arXiv:2604.03208 + arXiv:2506.07255 + arXiv:2504.04366 + arXiv:1011.0686 + arXiv:2604.11351 + arXiv:1706.04599)

flagged_for_v430: poe_world_factored_executable_subgoal_planner
(arXiv:2505.10819 + arXiv:2605.05138)

**Bottom line for .430:** make hierarchical subgoals the primary .430 input,
with DAgger/calibrated value routing as the affordable low-level guide; keep
PoE-World/factored executable planning as the stronger second candidate when
transition-factor trust is available. Do not re-open macro depth, off-centroid
click coverage, just-explore schedule extraction, or standalone goal-energy
heuristics.
{STUDYING_SECTION_END}
"""


def build_artifact(
    *,
    methods_mapped: Sequence[JsonMap] = DEFAULT_METHODS_MAPPED,
    citations_verified: JsonMap = CITATIONS_VERIFIED,
    preconditions_checked: JsonMap = DEFAULT_PRECONDITIONS_CHECKED,
    flagged_for_next_roadmap: Sequence[str] = FLAGGED_FOR_NEXT_ROADMAP,
    dead_levers_not_reflagged: Sequence[str] = DEAD_LEVERS_NOT_REFLAGGED,
    honest_verdict: str = HONEST_VERDICT,
) -> dict[str, object]:
    """Build and validate the REQ-ARC-WMTE-4661 mapping artifact."""

    artifact: dict[str, object] = {
        "honest_verdict": honest_verdict,
        "inference_substrate": INFERENCE_SUBSTRATE,
        "deep_research_not_used": True,
        "methods_mapped": [dict(method) for method in methods_mapped],
        "citations_verified": {
            source_id: dict(citation) for source_id, citation in citations_verified.items()
        },
        "flagged_for_next_roadmap": list(flagged_for_next_roadmap),
        "dead_levers_not_reflagged": list(dead_levers_not_reflagged),
        "note_path": NOTE_RELATIVE_PATH,
        "preconditions_checked": dict(preconditions_checked),
        "random_seed": RANDOM_SEED,
        "field_principles": dict(FIELD_PRINCIPLES),
    }
    validate_artifact(artifact)
    return artifact


def validate_artifact(artifact: JsonMap) -> None:
    """Validate the artifact so uncited generation-guidance claims fail closed."""

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
        raise ValueError("note_path must point at the 2026-06-24 generation-guidance note")
    if artifact["deep_research_not_used"] is not True:
        raise ValueError("deep_research_not_used must be true")
    if artifact["random_seed"] != RANDOM_SEED:
        raise ValueError("random_seed must be the bare integer 4661")

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
            or "A1 value-routing" not in mapping
            or "A2 energy-fitness QD" not in mapping
            or "live E3" not in mapping
        ):
            raise ValueError(
                "method mapping must name A1 value-routing, A2 energy-fitness QD, "
                "and live E3"
            )
        for field in REQUIRED_METHOD_FIELDS - {"source_ids"}:
            value = method[field]
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"method {field} must be a non-empty string")

    flagged = artifact["flagged_for_next_roadmap"]
    if any(
        token in item.lower()
        for item in flagged
        for token in DEAD_REFLAG_TOKENS
    ):
        raise ValueError("flagged_for_next_roadmap must not re-flag a dead lever")
    if flagged != FLAGGED_FOR_NEXT_ROADMAP or not all(
        "flagged_for_v430" in item for item in flagged
    ):
        raise ValueError("flagged_for_next_roadmap must name the .430 candidates")

    if artifact["dead_levers_not_reflagged"] != DEAD_LEVERS_NOT_REFLAGGED:
        raise ValueError("dead_levers_not_reflagged must name exactly the retired levers")

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
        raise ValueError("ops docs must not be modified by Exp 4661")


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
    """Check that the paired note maps verified sources to .430 work."""

    artifact_from_note(markdown)
    required_phrases = (
        "Fresh-pass provenance",
        "SOTA -> experiment mapping",
        "A1 value-routing",
        "A2 energy-fitness QD",
        "live E3",
        "Bottom line for the .430 roadmap",
        "flagged_for_v430",
        "Hierarchical Planning with Latent World Models",
        "PoE-World",
        "DAgger",
        "macro-action horizon-collapse RETIRED",
        "click-heatmap off-centroid generator RETIRED",
        "just-explore schedule-extraction CLOSED",
        "goal-energy heuristic NULL",
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
    return f"""# Generation-guidance SOTA ingestion 2026-06-24

```json
{artifact_json}
```

## Fresh-pass provenance

Read `AGENTS.md`, `CODEX.md`,
`results/experiment_4649_sota_ingestion_energy_fitness_generator.json`,
`docs/research-notes/energy-fitness-generator-literature-2026-06-23.md`,
`research-studying.md`, `research-references.md`,
`docs/research-notes/macro-vocab-prototype-finding-2026-06-23.md`,
`docs/research-notes/click-heatmap-generator-falsified-2026-06-23.md`,
`docs/research-notes/h2h-just-explore-vs-bare-explorer-2026-06-23.md`,
`results/experiment_4652_value_routing_cost_fix_live.json`, and
`results/experiment_4653_energy_fitness_qd_generation_live.json`. The current
stack is A1 value-routing plus A2 energy-fitness QD inside the live E3 agent
path; both returned no live lift, so this pass maps surviving SOTA directions
that can add generation guidance rather than re-ranking the same empty pool.

Reliable-channel pass, not `/deep-research`:
- `curl -sf -o /dev/null https://huggingface.co/api/models`
- `.venv/bin/python scripts/sweep_clusters.py --help`
- `.venv/bin/python scripts/sweep_clusters.py 5 --max-results 8`
- `.venv/bin/python scripts/sweep_clusters.py 6 --max-results 8`
- `.venv/bin/python scripts/sweep_semscholar.py - --limit 8` with three focused queries
- low-concurrency WebSearch/WebFetch of the top hierarchical, factored-model, and value-routing papers
- direct arXiv HTTP checks for all cited IDs

Semantic Scholar returned HTTP 429 for the focused queries and no S2-only source
was promoted. Direct arXiv HTTP checks returned 200 for arXiv:2604.03208,
arXiv:2506.07255, arXiv:2504.04366, arXiv:2505.10819, arXiv:2605.05138,
arXiv:1011.0686, arXiv:2604.11351, arXiv:1706.04599, arXiv:2102.04518,
arXiv:2605.28814, arXiv:2308.05483, and arXiv:2504.01915. The A2 current-stack
context is BES/QD action-sequence evolution from arXiv:2605.28814,
arXiv:2308.05483, and arXiv:2504.01915, but the live artifact did not generate
a winner. No live LLM inference, No training, No leaderboard submission, no
model load, and no live solve claim were run or made. `scripts/research_conductor.py`,
`ops/changelog.md`, and `ops/status.md` were not edited by this workflow.

Dead levers confirmed not re-flagged: macro-action horizon-collapse RETIRED;
click-heatmap off-centroid generator RETIRED; just-explore schedule-extraction
CLOSED; goal-energy heuristic NULL.

## SOTA -> experiment mapping

## Hierarchical subgoal search over the live E3 frontier

**Sources:** Hierarchical Planning with Latent World Models, arXiv:2604.03208;
Subgoal-Guided Policy Heuristic Search with Learned Subgoals, arXiv:2506.07255;
Sokoban hierarchical landmarks, arXiv:2504.04366; Executable World Models for
ARC-AGI-3, arXiv:2605.05138.

**Mapping to current stack:** A1 value-routing should stop acting as a global
ranker and instead serve as a calibrated tie-breaker inside each low-level
subgoal search. A2 energy-fitness QD should mutate sequences under a named
subgoal rather than evolve a broad standalone archive. The live E3 path remains
the only scored executor: every candidate path must replay under the existing
action and parity gates.

**Implementation cost over current stack:** medium. Add subgoal mining from
failed A1/A2 trees, run bounded low-level search for each selected subgoal, and
retain matched baseline, random/subgoal ablation, and replay gates.

**Fails when:** mined subgoals are only visually plausible, A1 remains
distribution-shifted on live frontier states, A2 burns budget near unreachable
landmarks, or live E3 rejects the replayed path.

## PoE-World factored executable model subgoal planner

**Sources:** PoE-World, arXiv:2505.10819; Executable World Models for
ARC-AGI-3, arXiv:2605.05138.

**Mapping to current stack:** A1 value-routing scores which expert-predicted
states deserve expansion; A2 energy-fitness QD mutates only sequences that the
factored model says are feasible; live E3 executes and audits the emitted plans.
This directly attacks the bridge gap left by the A2 null: QD needs a better
feasibility model before it mutates.

**Implementation cost over current stack:** medium-high. Induce object-level
precondition/effect experts, weight them by held-out transition trust, compose
only replay-stable experts, and search through the product model with hard live
replay checks.

**Fails when:** expert factors are not independent, rare object interactions
are smoothed away, generated experts overfit prefix transitions, or soft model
planning emits a live-invalid action sequence.

## Distribution-shift-corrected value routing for subgoal frontiers

**Sources:** DAgger, arXiv:1011.0686; WM-DAgger, arXiv:2604.11351; calibration,
arXiv:1706.04599; A* value heuristics, arXiv:2102.04518.

**Mapping to current stack:** A1 value-routing failed as a cost-fixed live lift,
which points to residual distribution shift or calibration. The repair is not a
bigger raw value weight. Aggregate the live frontier states where A1/A2 fail,
calibrate scores into bounded costs, and cache decision-point Q/value estimates.
A2 energy-fitness QD then receives subgoal costs instead of raw goal-energy
ranking, while live E3 keeps primitive actions and parity gates.

**Implementation cost over current stack:** medium. Add frontier-state logging,
DAgger-style aggregation or WM-DAgger rollouts, temperature/isotonic calibration
for cost deltas, and decision-point caching under the existing affordability
guard.

**Fails when:** aggregated frontier data is too small, hidden-state games break
calibration, cached values go stale after level-up, or routing only reorders a
candidate pool that still lacks the winning action.

## Bottom line for the .430 roadmap

1. Build `flagged_for_v430: hierarchical_subgoal_e3_frontier_with_distribution_shift_value_routing`
   first. It combines the strongest search structure, arXiv:2604.03208,
   arXiv:2506.07255, and arXiv:2504.04366, with the value-distribution repair
   from arXiv:1011.0686, arXiv:2604.11351, and arXiv:1706.04599.
2. Keep `flagged_for_v430: poe_world_factored_executable_subgoal_planner` as the
   second live level-up candidate. PoE-World arXiv:2505.10819 plus executable
   ARC world models arXiv:2605.05138 is the best factored-model answer to A2's
   no-winner bridge gap.
3. Treat raw BES/QD sources arXiv:2605.28814, arXiv:2308.05483, and
   arXiv:2504.01915 as current-stack context only until subgoal or factored-model
   guidance changes the candidate pool.
4. Do not re-open macro-action horizon-collapse RETIRED, click-heatmap
   off-centroid generator RETIRED, just-explore schedule-extraction CLOSED, or
   goal-energy heuristic NULL as `.430` inputs.
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
    """Write the default Exp 4661 deliverables under the repository root."""

    repo_root = Path(os.environ.get("CARNOT_EXP4661_ROOT", Path(__file__).resolve().parents[2]))
    artifact = write_outputs(
        artifact_path=repo_root / RESULT_RELATIVE_PATH,
        note_path=repo_root / NOTE_RELATIVE_PATH,
        studying_path=repo_root / "research-studying.md",
    )
    print(artifact["honest_verdict"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
